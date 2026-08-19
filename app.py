"""
TinyLLM ChatGPT-style server
- Serves chat.html as ChatGPT clone
- /api/chat with messages array
- /api/generate for raw completion
- /api/model_info
"""
import os
import json
import glob
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import mimetypes

from tokenizers import Tokenizer
from llm.config import TinyConfig
from llm.model import TinyLLM
from llm.tokenizer import TikTokenizerWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TOKENIZER = None
CFG = None
CKPT_PATH = None

def load():
    global MODEL, TOKENIZER, CFG, CKPT_PATH
    ckpt_path = "checkpoints/model_final.pt"
    candidates = glob.glob("checkpoints/model_*.pt")
    if not os.path.exists(ckpt_path) and candidates:
        ckpt_path = sorted(candidates, key=lambda x: os.path.getmtime(x))[-1]
    if not os.path.exists(ckpt_path):
        print("No checkpoint found - running random init model for demo")
        CFG = TinyConfig()
        MODEL = TinyLLM(CFG).to(DEVICE)
        if os.path.exists(CFG.tokenizer_path):
            hf = Tokenizer.from_file(CFG.tokenizer_path)
            TOKENIZER = TikTokenizerWrapper(hf)
            print(f"Random model loaded: {MODEL.count_params():,} params")
        return

    CKPT_PATH = ckpt_path
    print(f"Loading checkpoint {ckpt_path} on {DEVICE}")
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"Failed load {e}, trying cpu")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    CFG = ckpt.get('config', TinyConfig())
    MODEL = TinyLLM(CFG).to(DEVICE)
    MODEL.load_state_dict(ckpt['model'])
    MODEL.eval()
    hf = Tokenizer.from_file(CFG.tokenizer_path)
    TOKENIZER = TikTokenizerWrapper(hf)
    print(f"Model loaded: {MODEL.count_params():,} params from {ckpt_path}")

def build_chat_prompt(messages, tokenizer, max_context_tokens=200):
    """
    messages: list of {role, content}
    Build string: System: ... \nUser: ... \nAssistant: ...
    Truncate to fit context window
    """
    # Format
    parts = []
    for m in messages:
        role = m.get('role','user').lower()
        content = m.get('content','').strip()
        if role == 'system':
            parts.append(f"System: {content}")
        elif role == 'user':
            parts.append(f"User: {content}")
        elif role == 'assistant':
            parts.append(f"Assistant: {content}")
    # Add final assistant prefix to generate
    if not parts or not parts[-1].startswith("Assistant:"):
        parts.append("Assistant:")

    prompt = "\n".join(parts)

    # Truncate if too long: keep system + last few
    if tokenizer:
        ids = tokenizer.encode(prompt)
        if len(ids) > max_context_tokens:
            # Keep system message + last 2 exchanges
            # Try to keep last ~max_context_tokens
            # Simple: take tail tokens and decode back? For chat we want to keep recent.
            # We'll iteratively drop oldest non-system messages
            # Keep first system if exists
            system_msg = None
            rest = []
            for m in messages:
                if m.get('role')=='system' and system_msg is None:
                    system_msg = m
                else:
                    rest.append(m)
            # Drop from start of rest until fits
            while rest:
                test_parts = []
                if system_msg:
                    test_parts.append(f"System: {system_msg['content']}")
                for mm in rest:
                    r = mm.get('role','user')
                    if r=='user':
                        test_parts.append(f"User: {mm.get('content','')}")
                    elif r=='assistant':
                        test_parts.append(f"Assistant: {mm.get('content','')}")
                test_parts.append("Assistant:")
                test_prompt = "\n".join(test_parts)
                test_ids = tokenizer.encode(test_prompt)
                if len(test_ids) <= max_context_tokens:
                    prompt = test_prompt
                    break
                # drop oldest
                rest.pop(0)
            else:
                # if still too long, just truncate to tail tokens
                ids = tokenizer.encode(prompt)
                ids = ids[-max_context_tokens:]
                prompt = tokenizer.decode(ids)

    return prompt

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self.serve_file("chat.html", "text/html")
        elif path == "/chat" or path == "/chat.html":
            self.serve_file("chat.html", "text/html")
        elif path == "/llm_demo.html":
            self.serve_file("llm_demo.html", "text/html")
        elif path == "/game" or path == "/game.html":
            self.serve_file("index.html", "text/html")
        elif path == "/api/model_info":
            info = {
                "params": MODEL.count_params() if MODEL else 0,
                "checkpoint": CKPT_PATH or "random-init",
                "vocab": TOKENIZER.vocab_size() if TOKENIZER else 4096,
                "context": CFG.max_seq_len if CFG else 256,
                "device": str(DEVICE),
                "total_tokens": 20000000,
            }
            self.send_json(info)
        elif path.startswith("/checkpoints/") or path.startswith("/data/"):
            self.send_error(404)
        elif os.path.exists("."+path) and not ".." in path:
            mime, _ = mimetypes.guess_type(path)
            self.serve_file(path.lstrip("/"), mime or "application/octet-stream")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length>0 else b''
        if parsed.path == "/api/generate":
            self.handle_generate(body)
        elif parsed.path == "/api/chat":
            self.handle_chat(body)
        else:
            self.send_response(404)
            self.end_headers()

    def handle_generate(self, body):
        try:
            data = json.loads(body) if body else {}
            prompt = data.get("prompt", "Once upon a time")
            max_tokens = int(data.get("max_tokens", 100))
            temp = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 50))
            top_p = float(data.get("top_p", 0.92))

            if MODEL is None or TOKENIZER is None:
                self.send_json({"error": "Model not loaded - train first"}, 500)
                return

            ids = TOKENIZER.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
            text = TOKENIZER.decode(out[0].tolist())

            self.send_json({"prompt": prompt, "generated": text, "params": MODEL.count_params(), "checkpoint": CKPT_PATH})
        except Exception as e:
            import traceback; traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def handle_chat(self, body):
        try:
            data = json.loads(body) if body else {}
            messages = data.get("messages", [])
            if not messages:
                # fallback to prompt
                prompt = data.get("prompt", "")
                if prompt:
                    messages = [{"role":"user","content":prompt}]
                else:
                    self.send_json({"error":"No messages"}, 400)
                    return

            max_tokens = int(data.get("max_tokens", 120))
            temp = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 50))
            top_p = float(data.get("top_p", 0.92))

            if MODEL is None or TOKENIZER is None:
                self.send_json({"error":"Model not loaded"}, 500)
                return

            # Build prompt with chat format, ensure fits context
            max_context = (CFG.max_seq_len if CFG else 256) - max_tokens - 5
            max_context = max(50, max_context)
            chat_prompt = build_chat_prompt(messages, TOKENIZER, max_context_tokens=max_context)

            ids = TOKENIZER.encode(chat_prompt)
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
            full_text = TOKENIZER.decode(out[0].tolist())

            # Extract assistant response after last "Assistant:"
            # full_text contains prompt + generation, so get new part
            # Find last occurrence of "Assistant:" in full_text
            # Generation starts after that
            # Since chat_prompt ends with "Assistant:", the new generation is after it
            # So cut prompt's token length from output
            # But decoding may be ambiguous, so simplest: take full_text[len(chat_prompt):] or token-based
            # Use token ids length
            prompt_len = len(ids)
            generated_ids = out[0].tolist()[prompt_len:]
            generated_text = TOKENIZER.decode(generated_ids)

            # Stop at next role marker if model hallucinates
            # Stop at "\nUser:" or "\nSystem:" or "\nAssistant:" if appears again
            stop_markers = ["\nUser:", "\nSystem:", "\nAssistant:", "User:", "System:"]
            cut_at = len(generated_text)
            for marker in stop_markers:
                idx = generated_text.find(marker)
                if idx != -1 and idx < cut_at:
                    cut_at = idx
            generated_text = generated_text[:cut_at].strip()

            if not generated_text:
                # fallback to full_text tail
                generated_text = full_text[len(chat_prompt):].strip()
                for marker in stop_markers:
                    if marker in generated_text:
                        generated_text = generated_text.split(marker)[0].strip()

            self.send_json({
                "generated": generated_text,
                "full_prompt": chat_prompt,
                "full_text": full_text,
                "params": MODEL.count_params(),
                "checkpoint": CKPT_PATH,
                "tokens_used": len(ids)
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            self.send_json({"error": str(e)}, status=500)

    def serve_file(self, path, ctype):
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            # allow iframe preview
            self.send_header("X-Frame-Options", "ALLOWALL")
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
    load()
    port = int(os.environ.get("PORT", 8000))
    print(f"Serving TinyLLM ChatGPT-style demo on http://0.0.0.0:{port}")
    print(f" - Chat: http://localhost:{port}/ (chat.html)")
    print(f" - Playground: http://localhost:{port}/llm_demo.html")
    print(f" - Game: http://localhost:{port}/game")
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")
