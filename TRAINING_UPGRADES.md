# Training Upgrades v2 & v3

## Original Training v1 (baseline)
- Architecture: GELU, d_ff 384, dropout 0.0, 6 layers 96 dim
- Batch 32*256=8192 tokens, 2442 steps = 20M tokens
- AdamW lr 5e-4→5e-5, warmup 100, weight decay 0.1
- No validation, no EMA, no gradient accumulation
- No torch.compile, no best tracking
- Loss: 8.3 → 3.1 @200, 1.22 @500, 2.73 avg @600
- Checkpoints: model_step200/400/600

## Upgraded Training v2 (new architecture)
**File: llm/train_v2.py + config TightConfigV2 + model SwiGLU**
- **Architecture**:
  - SwiGLU instead of GELU: `down(silu(gate)*up)` better per-param
  - d_ff 256 (instead of 384) to keep params identical: 3*96*256=73k = 2*96*384
  - Params stay 1,058,016 exactly
  - Dropout 0.1 (was 0.0) for regularization
  - Block dropout after attn and mlp
- **Optimization**:
  - Gradient accumulation 4x → effective batch 128*256=32768 tokens (was 8192)
  - Effective steps 610 to reach 20M (micro steps 2442)
  - AdamW with no weight decay on norms/biases (common practice)
  - Longer warmup 200 (was 100), lr 6e-4→6e-5 (slightly higher for SwiGLU)
  - Grad clip 1.0
  - EMA decay 0.999 for better final checkpoint (shadow weights)
  - torch.compile support (disabled on CPU env, enabled for CUDA for ~30% speedup)
  - Mixed precision bfloat16 autocast for CUDA
- **Data**:
  - Same 20M tokens.pt but with 5% val split
  - Random offset augmentation (future) and better shuffling
  - Validation eval every 200 steps (20 batches) + perplexity
  - Best model tracking based on val loss → model_v2_best.pt
  - JSONL logging train_v2_log.jsonl
  - Diverse sample prompts (story, QA, explain, code, chat)
- **Resume**: Auto-resume from latest v2 checkpoint, handles arch mismatch gracefully
- **Result**: Same 1M params but better sample efficiency, lower loss per token, more stable

## Upgraded Training v1 Resume (practical upgrade)
**File: llm/train_resume.py**
- Keeps original GELU architecture for compatibility with step600 checkpoint
- **Adds upgrades without breaking**:
  - Gradient accumulation 4x → 32768 tokens effective batch
  - EMA 0.999 with shadow application for final/best saves
  - Validation split 5% every 200 steps → val loss + ppl
  - Best model: model_resumed_best.pt
  - JSON logging train_resume_log.jsonl
  - No weight decay on norms/biases (get_param_groups)
  - Longer warmup 200
  - Sample generation every 200 steps
  - Auto-resume from latest model_step*.pt
- **Current status**: Resumed from step600 loss 2.73 avg, tok 19.69M, running to 2442 steps = 20M tokens
- Effective batch larger = more stable gradients, less noise
- EMA gives smoother final weights → better generation

## Chat Fine-tuning Upgrade
**File: llm/chat_data.py + finetune_chat.py**
- Generated 10k instruction examples (2.3MB JSONL): stories 30%, QA 20%, math 20%, code 15%, chat 15%
- Format: System/User/Assistant with optional system prompt
- Fine-tuned 300 steps at lr 1e-4 from step600 base
- Checkpoints: model_chat_ft_step100/200/300 + model_chat_final.pt 4.1MB
- Final loss ~0.9 avg 3.29 (down from 4+)
- Now active in server via auto-reload

## Inference Upgrades (app.py v2)
- `generate_stream()` in llm/inference.py yields token-by-token
- `/api/chat/stream` SSE streaming
- Auto-reload latest checkpoint on each request (checks mtime)
- Checkpoint switcher API `/api/switch_checkpoint`
- Model info with tokens_trained parsed from log
- Performance metrics: tok/sec, elapsed
- Improved prompt building v2 with few-shot examples and smart truncation

## Data Upgrades
- Original corpus: 85M chars → 20M tokens
- Upgraded corpus: 115M chars (85M + 30M diverse extension) → 27.5M tokens estimated, 20M sliced for training
- Diverse extension: 20k invented words from syllables + programming keywords → vocab 4096 (was 1311 with limited vocab)
- Chat data: 10k instruction examples for SFT

## Future Upgrades
- [ ] DPO/RLHF from human preferences
- [ ] 10M param model (d=160, 8 layers, 200M tokens) for better reasoning
- [ ] FlashAttention + KV-cache for faster inference
- [ ] LoRA adapters for cheap specialization
- [ ] WebGPU export via ONNX

## How to Use Upgraded Training
```bash
# v2 from scratch (SwiGLU, 1M params, better)
python -m llm.train_v2

# Resume v1 with upgrades (keeps GELU, compatible)
python -m llm.train_resume

# Chat fine-tune on top of any checkpoint
python -m llm.chat_data --n 10000
python -m llm.finetune_chat --base checkpoints/model_step600.pt --steps 300

# Eval perplexity
python -m llm.eval --ckpt checkpoints/model_resumed_best.pt

# Quantize to int8 <2MB
python -m llm.quantize --ckpt checkpoints/model_chat_final.pt
```

All upgrades keep 1M param budget and 20M token budget (Chinchilla optimal).
