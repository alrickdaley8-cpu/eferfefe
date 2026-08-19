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
├── daemon.sh           # background supervisor: start/stop/status/logs, auto-restart
├── status.py           # progress, loss, throughput, ETA of the background run
├── serve.py            # streaming web server (stdlib http.server, SSE)
└── ui.py               # the single-page chat UI
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
ai/daemon.sh start                                         # runs everything in the background
python -m ai.sample --prompt "# file: README.md" --tokens 200
python -m ai.serve --port 8000                             # web playground
```

Resume an interrupted run with the same `python -m ai.train ...` command (`--resume auto` is
the default); add `--time-budget 3600` to stop cleanly after an hour.

Generated data (`ai/data/corpus/`, `*.bin`) and `ai/checkpoints/` are git-ignored.

## Stage 2 — making it answer questions

A 5M-parameter model has no room to memorise the world; what it *can* learn is a chat format
and how to answer **from a context passage handed to it at inference time**. So stage 2 is
instruction tuning + retrieval grounding:

| file | what it does |
|---|---|
| `ai/build_reasoning.py` | **142,500 chain-of-thought examples (~28M tokens)**: worked arithmetic, string/counting, grounded field lookups and two-package multi-hop comparisons |
| `ai/build_qa.py` | pulls metadata for 11,922 PyPI packages → `qa/knowledge.jsonl`, and generates **145,864 QA examples** (~18M tokens) → `qa/sft.jsonl` |
| `ai/finetune.py` | SFT on those examples, loss on answer tokens only → `checkpoints/sft.pt` |
| `ai/retrieve.py` | BM25-lite keyword search over the 11,922-package knowledge base (stdlib only) |
| `ai/chat.py` | retrieve → build prompt → generate (CLI + `Assistant` class) |
| `ai/serve.py` + `ai/ui.py` | streaming chat UI: visible thought process, model switcher, live training panel |

Chat format (plain text, so no new tokenizer symbols are needed):

```
### Question:
Context: black 26.5.1: The uncompromising code formatter. License: MIT. Requires Python >=3.10. …
Question: How do I install black?

### Answer:
Install it with pip:

pip install black<|endoftext|>
```

The example mix: grounded factual QA (what is X / install / version / license / Python support /
dependencies / author / homepage / import), closed-book variants of the most memorisable facts,
"which package should I use for …" recommendations, 20k arithmetic and string exercises, core
Python how-tos, and — importantly — **refusals**: when the retrieved context does not contain the
answer, the target response is "The context does not mention that, so I cannot answer it reliably."

```bash
python ai/build_qa.py --packages 12000            # knowledge base + SFT set
python -m ai.finetune --tokens 15000000           # ~1 h on 2 vCPU
python -m ai.chat "what license does requests use?"
python -m ai.serve --port 8000                    # chat UI
```

### Honest expectations

This is a 5M-parameter model — about 1/35,000th of a frontier model. It will not answer *any*
question. What it can realistically do after the full run: reply in the right format, pull facts
(license, version, dependencies, install command) out of a retrieved context, recommend a package
for a described task, handle small arithmetic/string tasks, and refuse when the context is empty.
Anything outside the knowledge base is a coin flip at best, which is exactly why the refusal
behaviour is trained in.

## Stage 3 — reasoning

Four layers of 256 dimensions cannot do multi-step reasoning *inside* the activations. The fix
that works at this scale is to make the model externalise its steps into tokens and then read its
own steps back — so every reasoning example is trained in this shape:

```
### Answer:
<think>
dependencies listed: pydantic, starlette, typing-extensions.
looking for pydantic: it is in the list.
</think>
Yes, fastapi depends on pydantic.
```

`ai/build_reasoning.py` generates 142,500 such examples from the knowledge base (no network):

* **arithmetic with carries** — column-by-column addition, split multiplication, subtraction,
  percentages, averages, sequence continuation, magnitude comparison
* **string/counting** — spell the word out, mark the positions, then count
* **grounded lookups** — "the context is about black. it says: License: MIT." → answer
  (this alone fixed the old failure where the model answered with the *wrong package name*)
* **multi-hop over two contexts** — same/different licence, which has more dependencies, shared
  dependencies, higher version, newer Python requirement, pick-the-right-package
* **checked refusals** — "scanning the context for download counts. no download figure is present."

Supporting changes:

* the retriever now builds a **numbered two-package context** (`(1) … (2) …`) when a question
  names two packages, matching the multi-hop training format
* **retrieval gating** — arithmetic and string questions skip retrieval entirely (a package blurb
  was derailing them), and the BM25 confidence floor rose from 1.5 to 8.0
* the streamer splits generation into a **reasoning channel and an answer channel** (partial
  `<think>` tags are never leaked), so the UI shows the steps live in their own block and the
  final answer in the bubble; `--think` does the same in the CLI
* **careful mode** in the UI (temperature 0.15, top-k 5) for arithmetic and lookups
* SFT budget raised to 25M tokens over the combined 56.9M-token instruction+reasoning pool

Early evidence from a 13-minute preview fine-tune (2.9M SFT tokens on a partially pretrained base):
grounded lookups are already correct — *"the context is about black. it says: License: MIT."* →
*"black is released under the MIT license."* — and arithmetic produces the right column-by-column
shape with wrong digits, which is what the full run is expected to fix.

## The web UI

`python -m ai.serve --port 8000` (the daemon starts it automatically) serves a single-page app:

* **streaming answers** over server-sent events — tokens appear as they are sampled
* **a visible thought process**, because there is no hidden magic to hide: the panel shows the
  retrieval query, the top-3 candidate packages with their BM25 scores, the exact context that was
  selected, the full prompt that went into the model (token count included), then live decoding —
  per-token confidence and the top-5 alternative tokens with their probabilities — and finally
  tokens/s and total latency
* **a model switcher** — every checkpoint in `ai/checkpoints/` is offered (chat / base-latest /
  base-best / frozen snapshots) with its stage, step and tokens-seen; switching is instant and the
  two most recent models stay cached in memory
* **decoding controls** — RAG grounding on/off, chat template on/off (raw completion mode),
  temperature, top-k, max tokens
* **a training panel** — stage, tokens, step, loss, throughput and ETA of the background run,
  refreshed every 15 s, plus a hot-reload of any checkpoint the trainer rewrites

The same chat client is embedded in the repository landing page (`index.html`, generated by
`ai/make_page.py`): `ai/ui.py` exports `CSS` / `HEADER` / `MAIN` / `JS`, the server composes them
into the standalone app at `/app`, and the landing page embeds the identical widget under
*Talk to it*. Served by `ai.serve` the widget talks to the same origin; opened as a static file it
detects that no API is reachable, explains how to start one, and accepts `?api=http://host:8000`.

API: `GET /models`, `GET /status`, `POST /chat`, `POST /chat/stream` (SSE: `thought` → `token` → `done`),
CORS-enabled.
The same trace is available in the terminal with `python -m ai.chat --think "..."`, and
`python -m ai.chat --list` prints the checkpoints.

## Inference speed

Generation used to re-run the whole prompt through the model for **every** token — 373 prompt
tokens re-encoded 70 times to write one answer. The decoder now keeps a **KV cache**:

* `Attention.forward` takes `past=(k, v)`, applies RoPE at the right absolute offset, concatenates
  the cached keys/values and switches to non-causal attention for single-token queries
* `GPT.forward(..., past=, use_cache=, last_only=)` threads the per-layer cache through and, during
  generation, projects only the final position to logits (skipping 8,192 × 511 wasted logits)
* the prompt is encoded once as a prefill, then each new token costs one 1-token forward pass
* generation runs under `torch.inference_mode()`

Measured on the same question and checkpoint (2 vCPU, while pretraining was running):

| configuration | throughput |
|---|---|
| before (full re-forward, 1 thread) | ~23 tok/s |
| **KV cache, 1 thread** | **134–187 tok/s** |
| KV cache, 2 threads | 75 tok/s |
| KV cache + dynamic int8 | 42 tok/s |

So the answer to "faster" was the cache, not more threads or quantisation — for matmuls this small
the extra thread sync and int8 pack/unpack cost more than they save, which is why inference defaults
to a single thread (`LM_THREADS`, or `--threads`). `LM_QUANTIZE=1` still exists if you want to try
int8 on different hardware. A full reasoning answer now takes **~0.4 s end to end** instead of ~3 s.

## Running it in the background

Training is managed by a detached supervisor (`setsid` + `nohup`), so it is independent of any
shell, terminal or agent session:

```bash
ai/daemon.sh start      # detach and run the whole pipeline: pretrain -> instruction tuning
ai/daemon.sh status     # progress bar, loss, throughput, ETA
ai/daemon.sh logs       # follow the log
ai/daemon.sh stop       # graceful: the trainer checkpoints, then exits
ai/daemon.sh restart
python -m ai.status --watch   # same status, refreshing every 30s
```

What the supervisor guarantees:

* **crash-proof** — if a stage dies (it was OOM-killed once when a data job ran alongside it),
  the daemon restarts it from the last checkpoint with exponential backoff (5s → 5min cap)
* **stop-proof** — SIGTERM/SIGINT are caught by the trainers, which checkpoint and exit 0;
  `stop` then `start` resumes at the exact step it left off
* **stage tracking** — `pretrain.done` / `sft.done` marker files mean a finished stage is never
  re-run, and the daemon walks pretrain → SFT → idle on its own

Trainer upgrades that make this safe:

| upgrade | effect |
|---|---|
| exact resume | step, optimiser state **and** the data-sampling RNG are checkpointed |
| best-checkpoint | `best.pt` tracks the lowest validation loss seen |
| heartbeat | `status.json` (stage/step/tokens/loss/tok-per-s/ETA) every log interval |
| mid-training mix | last 20% of pretraining blends in 15% instruction batches, so the base already knows the chat format before SFT |
| signal handling | clean checkpointed shutdown instead of losing up to 250 steps |
| vectorised batching | one fancy-index gather per batch instead of a Python loop; denormals flushed |
| SFT parity | the fine-tune stage has the same resume, status, signal and marker behaviour |

## Training curve

Live metrics are appended to `ai/checkpoints/log.jsonl` (loss, EMA loss, tokens/s, val loss
every 250 steps) and human-readable output to `ai/checkpoints/train.log`; the SFT stage logs to `sft_log.jsonl` / `sft.log`.

Current run: pretraining to 100M tokens, then instruction tuning on 15M SFT tokens, chained in
one background job (~8 h total on 2 vCPU).
