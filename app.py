"""
TinyLLM ChatGPT-style server v2 - Improvements
- Streaming /api/chat/stream via SSE
- Improved prompt building with few-shot chat examples
- Checkpoint selector, model info
- KV-cache-ready inference
- Auto-reload latest checkpoint
"""
import os
import json
import glob
import time
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import mimetypes
import threading

from tokenizers import Tokenizer
from llm.config import TinyConfig
from llm.model import TinyLLM
from llm.tokenizer import TikTokenizerWrapper
from llm.inference import generate_stream, count_tokens
try:
    from llm.retriever_v2 import retrieve_v2 as retrieve
    from llm.retriever import build_context_prompt
    from llm.retriever_v2 import FULL_KB
    print(f"[RAG] Using expanded retriever_v2 with {len(FULL_KB)} chunks for any-question answering")
except ImportError as e:
    print(f"[RAG] v2 not available, using v1: {e}")
    from llm.retriever import retrieve, build_context_prompt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TOKENIZER = None
CFG = None
CKPT_PATH = None
CKPT_MTIME = 0
LOCK = threading.Lock()

def load_latest():
    global MODEL, TOKENIZER, CFG, CKPT_PATH, CKPT_MTIME
    with LOCK:
        candidates = glob.glob("checkpoints/model_*.pt")
        if not candidates:
            # try final
            if os.path.exists("checkpoints/model_final.pt"):
                candidates = ["checkpoints/model_final.pt"]
            else:
                return False

        latest = max(candidates, key=lambda x: os.path.getmtime(x))
        mtime = os.path.getmtime(latest)
        if latest == CKPT_PATH and mtime == CKPT_MTIME and MODEL is not None:
            return True

        print(f"[Loader] Loading {latest} (mtime {mtime}) on {DEVICE}")
        try:
            ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        except Exception as e:
            print(f"[Loader] Failed {e}")
            return False

        cfg = ckpt.get('config', TinyConfig())
        model = TinyLLM(cfg).to(DEVICE)
        model.load_state_dict(ckpt['model'])
        model.eval()

        # tokenizer (should already exist)
        tok_path = cfg.tokenizer_path if hasattr(cfg,'tokenizer_path') else "checkpoints/tokenizer.json"
        if not os.path.exists(tok_path):
            tok_path = "checkpoints/tokenizer.json"
        hf = Tokenizer.from_file(tok_path)
        wrapper = TikTokenizerWrapper(hf)

        MODEL = model
        TOKENIZER = wrapper
        CFG = cfg
        CKPT_PATH = latest
        CKPT_MTIME = mtime
        print(f"[Loader] Loaded {latest}: {model.count_params():,} params, loss {ckpt.get('loss','?')}")
        return True

def build_chat_prompt_v2(messages, tokenizer, max_context_tokens=200, few_shot=True, use_rag=True):
    """
    Improved ChatML-like prompt with few-shot examples + RAG to answer any question
    Makes 1M model handle any question via retrieved knowledge injection
    Format: System: ... \n [RAG context] \n User: ... \n Assistant:
    """
    # Few-shot examples to make tiny model handle any question - diverse general knowledge
    few_shots = []
    if few_shot:
        few_shots = [
            {"role":"user","content":"What is the capital of France?"},
            {"role":"assistant","content":"The capital of France is Paris."},
            {"role":"user","content":"What is 2 + 2?"},
            {"role":"assistant","content":"2 + 2 = 4. Adding two and two gives four."},
            {"role":"user","content":"What is photosynthesis?"},
            {"role":"assistant","content":"Photosynthesis is how plants make food using sunlight, carbon dioxide, and water."},
            {"role":"user","content":"Tell me a short story about Lily."},
            {"role":"assistant","content":"Once upon a time, Lily went to the forest. She found a shiny key under a tree. The key opened a magic door to a garden full of flowers."},
        ]

    # RAG: retrieve relevant knowledge for last user message to make model answer any question
    rag_context = ""
    if use_rag and messages:
        last_user = None
        for m in reversed(messages):
            if m.get('role')=='user':
                last_user = m.get('content','')
                break
        if last_user:
            retrieved = retrieve(last_user, top_k=3)
            if retrieved:
                rag_context = build_context_prompt(retrieved)

    # Combine: system + few_shot + actual messages
    all_msgs = []
    system_msg = None
    for m in messages:
        if m.get('role')=='system' and system_msg is None:
            system_msg = m
        else:
            all_msgs.append(m)

    # Build with system first
    prompt_parts = []
    if system_msg:
        prompt_parts.append(f"System: {system_msg['content']}")

    # Add RAG context right after system to help answer any question
    # To fit 256 context, we need to be selective
    if rag_context:
        # Keep it concise to fit 256 context - truncate to first 2 facts if needed
        # Keep only first 200 chars of RAG to save tokens
        rag_short = rag_context.strip()[:400]
        prompt_parts.append(f"Context: {rag_short}")

    # Add few-shots after system but before real history (helps instruction following)
    # To fit context, if RAG is present, use only 1 few-shot example, else use 2
    if few_shot and len(messages) <= 3:
        # Choose subset to fit
        if rag_context:
            # Only 1 example when RAG present to save tokens
            subset = few_shots[:2]  # 1 Q/A pair = 2 items
        else:
            subset = few_shots[:4]  # 2 Q/A pairs
        for fs in subset:
            if fs['role']=='user':
                prompt_parts.append(f"User: {fs['content']}")
            else:
                prompt_parts.append(f"Assistant: {fs['content']}")

    # Add actual conversation (last N turns)
    # Keep last 4 exchanges max for 256 context
    # Truncate from start if needed
    for m in all_msgs:
        role = m.get('role','user')
        content = m.get('content','').strip()
        if role=='user':
            prompt_parts.append(f"User: {content}")
        elif role=='assistant':
            prompt_parts.append(f"Assistant: {content}")

    # Final assistant prefix
    if not prompt_parts or not prompt_parts[-1].startswith("Assistant:"):
        prompt_parts.append("Assistant:")

    prompt = "\n".join(prompt_parts)

    # Truncate to fit - preserve RAG and system
    if tokenizer:
        ids = tokenizer.encode(prompt)
        if len(ids) > max_context_tokens:
            # Keep RAG + system + last 2 turns (drop few-shot first to save tokens)
            keep = []
            if system_msg:
                keep.append(f"System: {system_msg['content']}")
            if rag_context:
                rag_short_trunc = rag_context.strip()[:200]  # even shorter for truncation case
                keep.append(f"Context: {rag_short_trunc}")
            # take last 2 messages from all_msgs (1 exchange) to minimize tokens
            last = all_msgs[-2:] if len(all_msgs)>2 else all_msgs
            for mm in last:
                if mm.get('role')=='user':
                    keep.append(f"User: {mm.get('content','')}")
                elif mm.get('role')=='assistant':
                    keep.append(f"Assistant: {mm.get('content','')}")
            keep.append("Assistant:")
            prompt = "\n".join(keep)
            # final hard cut if still too long
            ids = tokenizer.encode(prompt)
            if len(ids) > max_context_tokens:
                # hard cut to tail
                ids = ids[-max_context_tokens:]
                prompt = tokenizer.decode(ids)

    return prompt

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html" or path == "/chat" or path == "/chat.html":
            self.serve_file("chat.html", "text/html")
        elif path == "/llm_demo.html":
            self.serve_file("llm_demo.html", "text/html")
        elif path == "/game" or path == "/game.html":
            self.serve_file("index.html", "text/html")
        elif path == "/api/model_info":
            load_latest()
            info = {
                "params": MODEL.count_params() if MODEL else 0,
                "checkpoint": CKPT_PATH or "none",
                "checkpoints": sorted(glob.glob("checkpoints/model_*.pt"), key=lambda x: os.path.getmtime(x), reverse=True)[:10],
                "vocab": TOKENIZER.vocab_size() if TOKENIZER else 4096,
                "context": CFG.max_seq_len if CFG else 256,
                "device": str(DEVICE),
                "total_tokens": 20000000,
                "tokens_trained": self.get_trained_tokens(),
            }
            self.send_json(info)
        elif path == "/api/checkpoints":
            cps = sorted(glob.glob("checkpoints/model_*.pt"), key=lambda x: os.path.getmtime(x), reverse=True)
            data = [{"path": p, "name": os.path.basename(p), "size_mb": round(os.path.getsize(p)/1024/1024,2), "mtime": os.path.getmtime(p)} for p in cps]
            self.send_json(data)
        elif path.startswith("/checkpoints/") or path.startswith("/data/"):
            self.send_error(404)
        elif os.path.exists("."+path) and not ".." in path:
            mime, _ = mimetypes.guess_type(path)
            self.serve_file(path.lstrip("/"), mime or "application/octet-stream")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def get_trained_tokens(self):
        # parse from training log
        try:
            with open("/tmp/arena-workspace/procs/full-train-20m-8c461c27/out.log","rb") as f:
                f.seek(-2000, 2)
                tail = f.read().decode(errors='ignore')
                # find last tok= X.XXM
                import re
                m = re.findall(r'tok=([\d\.]+)M', tail)
                if m:
                    return float(m[-1])
        except:
            pass
        return 0

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length>0 else b''
        if parsed.path == "/api/generate":
            self.handle_generate(body)
        elif parsed.path == "/api/chat":
            self.handle_chat(body, stream=False)
        elif parsed.path == "/api/chat/stream":
            self.handle_chat(body, stream=True)
        elif parsed.path == "/api/switch_checkpoint":
            self.handle_switch(body)
        else:
            self.send_response(404)
            self.end_headers()

    def handle_switch(self, body):
        try:
            data = json.loads(body)
            ckpt = data.get("checkpoint")
            if ckpt and os.path.exists(ckpt):
                global CKPT_PATH, CKPT_MTIME
                CKPT_PATH = None
                CKPT_MTIME = 0
                # force load specific
                with LOCK:
                    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
                    cfg = ck.get('config', TinyConfig())
                    model = TinyLLM(cfg).to(DEVICE)
                    model.load_state_dict(ck['model'])
                    model.eval()
                    hf = Tokenizer.from_file(cfg.tokenizer_path if hasattr(cfg,'tokenizer_path') else "checkpoints/tokenizer.json")
                    wrapper = TikTokenizerWrapper(hf)
                    globals()['MODEL'] = model
                    globals()['TOKENIZER'] = wrapper
                    globals()['CFG'] = cfg
                    globals()['CKPT_PATH'] = ckpt
                    globals()['CKPT_MTIME'] = os.path.getmtime(ckpt)
                self.send_json({"ok":True, "checkpoint": ckpt, "params": model.count_params()})
            else:
                self.send_json({"error":"Checkpoint not found"}, 404)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.send_json({"error":str(e)},500)

    def handle_generate(self, body):
        try:
            load_latest()
            data = json.loads(body) if body else {}
            prompt = data.get("prompt", "Once upon a time")
            max_tokens = int(data.get("max_tokens", 100))
            temp = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 50))
            top_p = float(data.get("top_p", 0.92))

            if MODEL is None or TOKENIZER is None:
                self.send_json({"error": "Model not loaded"}, 500)
                return

            ids = TOKENIZER.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            start = time.time()
            with torch.no_grad():
                out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
            elapsed = time.time()-start
            text = TOKENIZER.decode(out[0].tolist())
            tok_per_sec = max_tokens/elapsed if elapsed>0 else 0
            self.send_json({"prompt": prompt, "generated": text, "params": MODEL.count_params(), "checkpoint": CKPT_PATH, "elapsed": elapsed, "tok_per_sec": tok_per_sec})
        except Exception as e:
            import traceback; traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def handle_chat(self, body, stream=False):
        try:
            load_latest()
            data = json.loads(body) if body else {}
            messages = data.get("messages", [])
            if not messages:
                prompt = data.get("prompt", "")
                if prompt:
                    messages = [{"role":"user","content":prompt}]
                else:
                    self.send_json({"error":"No messages"}, 400)
                    return

            max_tokens = int(data.get("max_tokens", 150))
            temp = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 50))
            top_p = float(data.get("top_p", 0.92))

            if MODEL is None or TOKENIZER is None:
                self.send_json({"error":"Model not loaded"}, 500)
                return

            max_context = (CFG.max_seq_len if CFG else 256) - max_tokens - 5
            max_context = max(60, max_context)
            chat_prompt = build_chat_prompt_v2(messages, TOKENIZER, max_context_tokens=max_context, few_shot=True)

            prompt_ids = TOKENIZER.encode(chat_prompt)

            if not stream:
                # Non-streaming, fast
                x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
                start = time.time()
                with torch.no_grad():
                    out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
                elapsed = time.time()-start
                full_text = TOKENIZER.decode(out[0].tolist())
                # extract new
                generated_ids = out[0].tolist()[len(prompt_ids):]
                generated_text = TOKENIZER.decode(generated_ids)
                # cut stop markers
                for marker in ["\nUser:", "\nSystem:", "\nAssistant:", "User:", "System:"]:
                    if marker in generated_text:
                        generated_text = generated_text.split(marker)[0]
                generated_text = generated_text.strip()

                self.send_json({
                    "generated": generated_text,
                    "full_prompt": chat_prompt,
                    "full_text": full_text,
                    "params": MODEL.count_params(),
                    "checkpoint": CKPT_PATH,
                    "tokens_used": len(prompt_ids),
                    "elapsed": elapsed,
                    "tok_per_sec": len(generated_ids)/elapsed if elapsed>0 else 0
                })
            else:
                # Streaming via SSE
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                start = time.time()
                token_count = 0
                accumulated = ""

                # We need to stream incremental
                # Use generate_stream generator
                for token_id, full_decoded, done in generate_stream(MODEL, TOKENIZER, prompt_ids, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, top_p=top_p, device=DEVICE):
                    token_count += 1
                    # Extract only new assistant part
                    # full_decoded contains prompt + generated
                    prompt_text = TOKENIZER.decode(prompt_ids)
                    # new part
                    if len(full_decoded) > len(chat_prompt):
                        new_part = full_decoded[len(chat_prompt):]
                    else:
                        new_part = full_decoded

                    # Cut stop markers
                    truncated = False
                    for marker in ["\nUser:", "\nSystem:"]:
                        if marker in new_part:
                            new_part = new_part.split(marker)[0]
                            truncated = True
                            break

                    accumulated = new_part

                    # Send SSE event
                    payload = json.dumps({"token": TOKENIZER.decode([token_id]), "text": accumulated, "done": done or truncated, "tokens_used": len(prompt_ids)})
                    try:
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                    except BrokenPipeError:
                        break

                    if done or truncated:
                        break

                # final done event
                elapsed = time.time()-start
                final_payload = json.dumps({"generated": accumulated, "done": True, "elapsed": elapsed, "tok_per_sec": token_count/elapsed if elapsed>0 else 0, "checkpoint": CKPT_PATH, "params": MODEL.count_params()})
                try:
                    self.wfile.write(f"data: {final_payload}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except:
                    pass

        except Exception as e:
            import traceback; traceback.print_exc()
            if not stream:
                self.send_json({"error": str(e)}, 500)
            else:
                try:
                    self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                except:
                    pass

    def serve_file(self, path, ctype):
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Frame-Options", "ALLOWALL")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found: "+path.encode())

    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format%args))

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    load_latest()
    port = int(os.environ.get("PORT", 8000))
    print(f"Serving TinyLLM v2 ChatGPT-style on http://0.0.0.0:{port}")
    print(f" - Chat: http://localhost:{port}/")
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")
