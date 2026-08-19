# tiny-lm — a 5M-parameter language model trained on 100M tokens

A complete, from-scratch LLM pipeline that runs inside this sandbox: corpus collection →
BPE tokenizer → 5.0M-parameter transformer → 100M-token training run → sampling CLI → web playground.

```
ai/
├── build_corpus.py     # download + clean ~470 MB of text from PyPI source releases
├── train_tokenizer.py  # byte-level BPE, vocab 8192
├── prepare_data.py     # tokenize corpus -> train.bin (100,000,000 tokens) + val.bin (1M)
├── model.py            # Llama-style decoder: RMSNorm, RoPE, SwiGLU, tied embeddings
├── train.py            # AdamW + cosine schedule, resumable checkpoints
├── sample.py           # text generation CLI
└── serve.py            # tiny web playground (stdlib http.server)
```

## The model — 5,015,808 parameters

| | |
|---|---|
| vocab size | 8,192 (byte-level BPE, `<\|endoftext\|>` = id 0) |
| layers | 4 |
| d_model | 256 |
| heads | 8 (head dim 32) |
| FFN | SwiGLU, hidden 608 |
| context | 512 tokens |
| position | RoPE (θ=10,000) |
| norm | RMSNorm (pre-norm) |
| head | tied to the token embedding |
| **total params** | **5,015,808** (2.92M non-embedding) |

Breakdown: embedding 2,097,152 + 4 × 729,088 per block + 256 final norm.

## The data — 100,000,000 tokens

The sandbox can only reach PyPI and the GitHub API, so the corpus is built from PyPI source
releases: 1,745 packages, 58,395 documents, 470 MB of text — a mix of English prose (READMEs,
docs, docstrings) and Python code. Documents are exact-dedup'd by hash, filtered for
printability / line-length / line-uniqueness, capped at 1.2 MB per package for diversity, and
separated by `<|endoftext|>`.

Tokenized with the 8k BPE tokenizer, the training set is truncated to **exactly 100,000,000
tokens** (191 MB of uint16) plus a held-out 1M-token validation split.
That is a 20:1 token/parameter ratio — Chinchilla-optimal territory.

## Training recipe

| | |
|---|---|
| optimizer | AdamW (β=0.9/0.95, wd 0.1 on matrices only) |
| LR | 1e-3, 200-step linear warmup → cosine → 1e-4 |
| batch | 16 × 512 = 8,192 tokens/step |
| steps | 12,207 (= 100M tokens, exactly 1 epoch) |
| grad clip | 1.0 |
| precision | fp32 (this CPU has no AVX512-BF16; bf16 measured 2.8× *slower*) |

Hardware here is 2 vCPU, so throughput is ~3,700 tokens/s ⇒ the full 100M-token run takes
about 7.5 hours of wall clock. The run checkpoints every 250 steps to `ai/checkpoints/` and
resumes automatically, so it can be stopped and restarted freely.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install torch numpy tokenizers requests

python ai/build_corpus.py --target-mb 450 --workers 24     # ~2 min, 470 MB
python ai/train_tokenizer.py --vocab-size 8192             # ~40 s
python ai/prepare_data.py --max-tokens 100000000           # ~95 s
python -m ai.train --total-tokens 100000000                # the run itself
python -m ai.sample --prompt "# file: README.md" --tokens 200
python -m ai.serve --port 8000                             # web playground
```

Resume an interrupted run with the same `python -m ai.train ...` command (`--resume auto` is
the default); add `--time-budget 3600` to stop cleanly after an hour.

Generated data (`ai/data/corpus/`, `*.bin`) and `ai/checkpoints/` are git-ignored.

## Training curve

Live metrics are appended to `ai/checkpoints/log.jsonl` (loss, EMA loss, tokens/s, val loss
every 250 steps) and human-readable output to `ai/checkpoints/train.log`.
