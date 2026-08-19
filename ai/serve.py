"""Tiny web playground for the 5M-parameter model (stdlib only, no extra deps).

    python -m ai.serve --port 8000
Then open the preview and type a prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ai.sample import load

MODEL = None
TOK = None
STEP = 0
LOCK = threading.Lock()

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "model.pt")
_mtime = 0.0


def maybe_reload() -> None:
    """Hot-swap in the newest checkpoint while a training run is still going."""
    global MODEL, TOK, STEP, _mtime
    try:
        m = os.path.getmtime(CKPT_PATH)
    except OSError:
        return
    if m > _mtime:
        with LOCK:
            MODEL, TOK, STEP = load(CKPT_PATH)
            _mtime = m
            print(f"[serve] loaded checkpoint step {STEP}", flush=True)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>tiny-lm playground</title>
<style>
 body{background:#0e1116;color:#e6edf3;font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
      margin:0;padding:32px;display:flex;justify-content:center}
 .wrap{width:min(860px,100%)}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#8b949e;font-size:13px;margin-bottom:20px}
 textarea{width:100%;height:110px;background:#161b22;color:#e6edf3;border:1px solid #30363d;
          border-radius:8px;padding:12px;font:inherit;resize:vertical}
 .row{display:flex;gap:12px;align-items:center;margin:12px 0}
 label{color:#8b949e;font-size:13px} input[type=range]{width:120px}
 button{background:#238636;color:#fff;border:0;border-radius:8px;padding:10px 18px;
        font:inherit;cursor:pointer} button:disabled{opacity:.5;cursor:wait}
 pre{white-space:pre-wrap;background:#161b22;border:1px solid #30363d;border-radius:8px;
     padding:14px;min-height:120px;overflow-x:auto}
</style></head><body><div class="wrap">
<h1>tiny-lm &middot; 5.0M parameters</h1>
<div class="sub" id="meta">loading…</div>
<textarea id="p"># file: README.md
# awesome-project
</textarea>
<div class="row">
  <label>tokens <input id="n" type="range" min="16" max="400" value="160"
    oninput="nv.textContent=this.value"></label><span id="nv">160</span>
  <label>temp <input id="t" type="range" min="10" max="150" value="80"
    oninput="tv.textContent=(this.value/100).toFixed(2)"></label><span id="tv">0.80</span>
  <button id="go" onclick="gen()">Generate</button>
</div>
<pre id="out"></pre>
</div><script>
fetch('/info').then(r=>r.json()).then(d=>{
  meta.textContent = d.params.toLocaleString()+' params · trained '+d.tokens_seen.toLocaleString()+
    ' tokens (step '+d.step+') · vocab '+d.vocab;});
async function gen(){
  go.disabled=true; out.textContent='generating…';
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:p.value,tokens:+n.value,temperature:+t.value/100})});
  const d=await r.json(); out.textContent=d.text; go.disabled=false;
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/info":
            maybe_reload()
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send(200, json.dumps({
                "params": MODEL.num_params(), "step": STEP,
                "tokens_seen": STEP * 8192, "vocab": MODEL.cfg.vocab_size}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        maybe_reload()
        if self.path != "/generate":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = req.get("prompt", "")
        eot = TOK.token_to_id("<|endoftext|>")
        ids = TOK.encode(prompt).ids or [eot]
        x = torch.tensor([ids[-MODEL.cfg.block_size:]], dtype=torch.long)
        with LOCK, torch.no_grad():
            out = MODEL.generate(x, max_new_tokens=int(req.get("tokens", 160)),
                                 temperature=float(req.get("temperature", 0.8)),
                                 top_k=40, top_p=0.95, eot_id=None)
        self._send(200, json.dumps({"text": TOK.decode(out[0].tolist(),
                                                       skip_special_tokens=False)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    global MODEL, TOK, STEP
    MODEL, TOK, STEP = load(args.ckpt) if args.ckpt else load()
    print(f"[serve] model step {STEP}, {MODEL.num_params():,} params on http://0.0.0.0:{args.port}",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
