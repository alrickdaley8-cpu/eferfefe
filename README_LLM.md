# TinyLLM — 1M Parameter AI trained on 20M Tokens

A from-scratch decoder-only Transformer (1.06M parameters) trained on 20M tokens (Chinchilla-optimal 20 tokens/param for 1M model). No pretrained weights, full pipeline included.

## Architecture (≈1.06M params)

- **Vocab**: 4096 BPE (ByteLevel, GPT-2 style)
- **d_model**: 96
- **Layers**: 6
- **Heads**: 6 (head_dim=16)
- **FF**: 384 (4×)
- **Positional**: RoPE (no learned pos emb → saves params)
- **Norm**: RMSNorm
- **Tied weights**: embedding ↔ lm_head

**Param count**:
- token emb: 4096×96 = 393,216
- 6 layers × (qkv 27,648 + out 9,216 + ff 73,728) = 663,552
- norms ≈ 1,344
- **Total ≈ 1,058,112** (with RoPE) — perfect for 1M target

Context length: 256 tokens.

## Training Budget: 20M tokens

Per Chinchilla scaling laws, optimal tokens ≈ 20× parameters.
- 1M params × 20 = 20M tokens

Training setup:
- Batch: 32 seqs × 256 tokens = 8192 tok/batch
- Steps: 20,000,000 / 8192 ≈ 2442 steps
- LR: 5e-4 → 5e-5 cosine with 100 warmup
- Optimizer: AdamW (β1=0.9, β2=0.95, wd=0.1)
- Loss: cross-entropy, causal LM
- Grad clip 1.0

## Dataset — 20M tokens (~85M chars)

Synthetically generated diverse corpus to reach 20M tokens offline (no HF download needed):

- 70% TinyStories-style child stories (N names, places, etc.)
- 10% QA / factual常识
- 10% Math reasoning (a+b, etc.)
- 5% Python code
- 5% Dialogues

Generator: `llm/build_corpus.py` → `data/corpus.txt` ~85MB, ~20M tokens after BPE.

You can also swap in your own text: put any .txt in `data/corpus.txt`. Tokenizer will adapt.

## File Layout

```
llm/
  config.py       # TinyConfig ~1M params, 20M token budget
  model.py        # Transformer with RoPE, RMSNorm
  tokenizer.py    # BPE training via HF tokenizers
  dataset.py      # streaming TextDataset, token counting
  build_corpus.py # 20M token corpus generator
  train.py        # training loop 2442 steps
  generate.py     # inference
checkpoints/
  tokenizer.json  # 4096 vocab
  model_step*.pt
  model_final.pt
data/
  corpus.txt      # ~85M chars
```

## Quick Start

```bash
# setup venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Build 20M token corpus (85M chars)
python -m llm.build_corpus --out data/corpus.txt --chars 85000000

# 2. Train tokenizer (4096 vocab)
python -m llm.tokenizer --corpus data/corpus.txt --out checkpoints/tokenizer.json --vocab 4096

# 3. Train model (20M tokens, ~1 hour CPU, ~5 min GPU)
python -m llm.train

# 4. Generate
python -m llm.generate --ckpt checkpoints/model_final.pt --prompt "Once upon a time, Lily" --tokens 150
```

## Demo Web UI

- `llm_demo.html` → chat with the model (requires backend `app.py`)
- Or run `python app.py` to serve inference server.

## Sample Outputs (after full training)

Prompt: "Once upon a time, Lily"
> Once upon a time, Lily went to the forest. She saw a brave little fox. Lily said hello, and the fox gave her a shiny key. Lily used the key to open a magic door...

Prompt: "Q: What is 12 + 35? A:"
> Q: What is 12 + 35? A: 12 + 35 = 47. Because adding them together gives 47.

## Why 1M / 20M?

- Shows full LLM pipeline minimal viable.
- Can train on laptop CPU in <2h, GPU in minutes.
- Chinchilla-optimal: 20 tok/param is compute-optimal frontier.
- Enough to memorize simple stories, add numbers, write code snippets.

## Extensions

- Increase to 10M params / 200M tokens for better reasoning.
- Context 1024, use FlashAttention.
- Add instruction tuning (`chat` format).
- Quantize to int8 for edge deployment (model <5MB).

---

Built from scratch for `alrickdaley8-cpu/eferfefe` arena.
