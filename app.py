"""
Simple FastAPI/Flask-like inference server for TinyLLM + serves demo UI
Uses only stdlib http.server + torch, no extra deps
"""
import os
import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes

from tokenizers import Tokenizer
from llm.config import TinyConfig
from llm.model import TinyLLM
from llm.tokenizer import TikTokenizerWrapper

# Load model once
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TOKENIZER = None
CFG = None

def load():
    global MODEL, TOKENIZER, CFG
    # find checkpoint
    ckpt_path = "checkpoints/model_final.pt"
    if not os.path.exists(ckpt_path):
        import glob
        cands = glob.glob("checkpoints/model_*.pt")
        if cands:
            ckpt_path = sorted(cands)[-1]
        else:
            print("No checkpoint found - running untrained model for demo")
            CFG = TinyConfig()
            MODEL = TinyLLM(CFG).to(DEVICE)
            # need tokenizer
            if os.path.exists(CFG.tokenizer_path):
                hf = Tokenizer.from_file(CFG.tokenizer_path)
                TOKENIZER = TikTokenizerWrapper(hf)
            else:
                print("No tokenizer found")
            return

    print(f"Loading checkpoint {ckpt_path} on {DEVICE}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    CFG = ckpt.get('config', TinyConfig())
    MODEL = TinyLLM(CFG).to(DEVICE)
    MODEL.load_state_dict(ckpt['model'])
    MODEL.eval()
    hf = Tokenizer.from_file(CFG.tokenizer_path)
    TOKENIZER = TikTokenizerWrapper(hf)
    print(f"Model loaded: {MODEL.count_params():,} params")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            # serve game original? serve demo selector
            self.serve_file("llm_demo.html", "text/html")
        elif parsed.path == "/llm_demo.html":
            self.serve_file("llm_demo.html", "text/html")
        elif parsed.path == "/game" or parsed.path == "/game.html":
            self.serve_file("index.html", "text/html")
        elif parsed.path.startswith("/checkpoints/") or parsed.path.startswith("/data/"):
            self.send_error(404)
        elif os.path.exists("."+parsed.path) and not ".." in parsed.path:
            # static files
            mime, _ = mimetypes.guess_type(parsed.path)
            self.serve_file(parsed.path.lstrip("/"), mime or "application/octet-stream")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                prompt = data.get("prompt", "Once upon a time")
                max_tokens = int(data.get("max_tokens", 100))
                temp = float(data.get("temperature", 0.8))
                top_k = int(data.get("top_k", 50))
                # generate
                if MODEL is None or TOKENIZER is None:
                    self.send_json({"error": "Model not loaded - train first"})
                    return
                ids = TOKENIZER.encode(prompt)
                x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, top_p=0.92)
                text = TOKENIZER.decode(out[0].tolist())
                self.send_json({"prompt": prompt, "generated": text, "params": MODEL.count_params()})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json({"error": str(e)}, status=500)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_file(self, path, ctype):
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")

    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # quiet
        print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format%args))

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    load()
    port = int(os.environ.get("PORT", 8000))
    print(f"Serving TinyLLM demo on http://0.0.0.0:{port}")
    print(f" - LLM demo: http://localhost:{port}/llm_demo.html")
    print(f" - Original game: http://localhost:{port}/game")
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")
