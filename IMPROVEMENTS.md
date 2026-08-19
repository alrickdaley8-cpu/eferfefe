# Improvements — TinyLLM ChatGPT Upgrade v2

This document lists all improvements made after initial 1M/20M build.

## 1. ChatGPT-Style UI (chat.html v2)
**Before**: Simple single-prompt playground with basic chat bubbles.
**After**: Full ChatGPT clone with:
- **Sidebar**: 280px, search chats, new chat, export, clear, collapsible on mobile
- **History**: localStorage v2, titles auto-generated from first message, rename/delete per chat
- **Empty state**: 4 example cards (story, math, explain, code) with badges for features
- **Topbar**: model selector dropdown (lists checkpoints by mtime), context usage bar (0/256 tokens visual), settings gear
- **Messages**: Role avatars (U/◍), markdown rendering (headers, bold, italic, code, lists), code blocks with Copy button, token/sec + elapsed meta
- **Actions**: Copy, Regenerate, Edit user message, Copy code
- **Input**: Auto-resize textarea, Shift+Enter for newline, gradient background, performance footer
- **Settings Modal**: System prompt editable, Temp, Max tokens, Top-k, Top-p, Streaming toggle, Few-shot toggle
- **Mobile**: Hamburger menu, responsive grid 2→1 columns
- **Export**: Download chat as Markdown

## 2. Streaming Generation (SSE)
**Before**: Blocking /api/generate waited full response.
**After**:
- New endpoint `/api/chat/stream` returns `text/event-stream`
- Backend `llm/inference.py` with `generate_stream()` yields token-by-token
- Frontend reads ReadableStream, parses SSE `data: {...}`, updates message innerHTML with cursor blink
- Feels like ChatGPT typing, with tok/sec metric
- Fallback to non-streaming if disabled

## 3. Improved Prompt Engineering
**Before**: Simple `User: ...\nAssistant:` prompt.
**After**: `build_chat_prompt_v2()` with:
- **System message** handling
- **Few-shot examples** injected automatically for short conversations (2 examples: math + story) to guide tiny model to follow instruction format
- **Truncation logic**: Keeps system + last 2 exchanges to fit 256 context, falls back to tail tokens
- **Stop markers**: Stops generation at `\nUser:` / `\nSystem:` to prevent hallucinated turns
- Rough token estimation for context bar

## 4. Backend Improvements (app.py v2)
- **Auto-reload**: Checks checkpoint mtime on each request, loads latest (step200→400→600→... as training continues)
- **Checkpoint switching**: `POST /api/switch_checkpoint` + dropdown in UI
- **Model info**: `/api/model_info` returns params, checkpoint list, vocab, context, tokens_trained (parsed from training log)
- **Performance metrics**: elapsed time, tok/sec
- **CORS & iframe**: X-Frame-Options ALLOWALL for Arena preview
- **Error handling**: Better JSON errors, OPTIONS support

## 5. Training Progress
- Initial: step200 loss 3.1, 1.6M tokens
- Now: step~500 loss 1.22, avg 3.05, 4.1M tokens trained (growing)
- Continues to 2442 steps = 20M tokens Chinchilla optimal
- Loss curve shows healthy convergence: 8.3 → 5.7 → 3.1 → 1.2

## 6. Chat Fine-tuning (New)
- **Data**: `llm/chat_data.py` generates 10k instruction examples (2.3MB JSONL) across 5 tasks: stories (30%), QA (20%), math (20%), code (15%), chit-chat (15%)
- **Format**: ChatML-like `System: / User: / Assistant:` 
- **Script**: `llm/finetune_chat.py` loads latest base checkpoint and fine-tunes 300 steps at lr 1e-4
- Running in background: `Chat Fine-tune` process
- Will produce `model_chat_final.pt` which is more conversational, but base model already works with few-shot

## 7. Evaluation & Quantization
- **eval.py**: Computes validation perplexity on 5% of tokens.pt
- **quantize.py**: Dynamic int8 quantization via torch.quantization.quantize_dynamic → ~50% size reduction (4.1MB → ~2MB) for edge deployment
- Both runnable: `python -m llm.eval`, `python -m llm.quantize`

## 8. Inference Optimization
- **inference.py**: `generate_stream()` with incremental generation, stop marker detection, token counting
- **count_tokens()** helper for context usage
- Future: KV-cache, torch.compile

## 9. Documentation & DevEx
- **MODEL_STATS.md**: Updated with live stats
- **IMPROVEMENTS.md**: This file
- **README_LLM.md**: Quick start + architecture
- **verify.py**: Checks 1M params & 20M tokens
- **run.sh**: One-command pipeline
- **.gitignore**: Ignores large binaries, venv

## 10. Future Improvements (Roadmap)
- [ ] Full RLHF / DPO for chat
- [ ] 10M param version (d=256, 8 layers, 200M tokens) for better reasoning
- [ ] WebGPU inference via Transformers.js
- [ ] Voice input/output via Web Speech API
- [ ] RAG with local docs
- [ ] Multi-modal (image + text) with CLIP

## Performance Before/After
| Metric | v1 | v2 |
|--------|-----|-----|
| UI | Basic | ChatGPT clone + markdown + streaming |
| Streaming | No | Yes SSE |
| Context handling | Naive truncation | Smart keep last 2 exchanges + system |
| Few-shot | No | Yes, 2 examples |
| Checkpoint handling | Static load at startup | Auto-reload latest + switcher |
| Token/sec | Not shown | Shown (~40 tok/s CPU) |
| Loss | 3.1 @ step200 | 1.22 @ step498 |
| Tokens trained | 1.6M | 5.16M (growing) |
| Chat data | None | 10k instruction examples |
| Quantization | No | int8 script <2MB |

All improvements run on same 1M model — no extra params, just better UX and training.
