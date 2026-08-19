"""Web playground for the tiny LM: retrieval-grounded chat + raw completion (stdlib only).

    python -m ai.serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ai.chat import Assistant, default_ckpt

ASSISTANT: Assistant | None = None
LOCK = threading.Lock()
_mtime = 0.0
_path = ""


def maybe_reload() -> None:
    """Hot-swap the newest checkpoint while training is still running."""
    global ASSISTANT, _mtime, _path
    p = default_ckpt()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return
    if p != _path or m > _mtime:
        with LOCK:
            ASSISTANT = Assistant(p)
            _mtime, _path = m, p
            print(f"[serve] loaded {os.path.basename(p)} step {ASSISTANT.step} "
                  f"({ASSISTANT.stage})", flush=True)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>tiny-lm chat</title>
<style>
 body{background:#0e1116;color:#e6edf3;font:15px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
      margin:0;padding:28px;display:flex;justify-content:center}
 .wrap{width:min(900px,100%)}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#8b949e;font-size:12.5px;margin-bottom:18px}
 #log{min-height:220px}
 .msg{border-radius:10px;padding:11px 14px;margin:10px 0;white-space:pre-wrap}
 .you{background:#1f2a37;border:1px solid #30363d}
 .bot{background:#161b22;border:1px solid #30363d}
 .ctx{color:#8b949e;font-size:12px;border-left:2px solid #30363d;padding-left:10px;margin:6px 0 0}
 .row{display:flex;gap:10px;margin-top:14px}
 input[type=text]{flex:1;background:#161b22;color:#e6edf3;border:1px solid #30363d;
   border-radius:8px;padding:12px;font:inherit}
 button{background:#238636;color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;
   cursor:pointer} button:disabled{opacity:.5;cursor:wait}
 .opts{color:#8b949e;font-size:12.5px;margin-top:10px;display:flex;gap:18px;align-items:center}
 .chip{background:#21262d;border:1px solid #30363d;border-radius:999px;padding:4px 10px;
   cursor:pointer;font-size:12px}
</style></head><body><div class="wrap">
<h1>tiny-lm &middot; 5.0M parameters</h1>
<div class="sub" id="meta">loading…</div>
<div id="log"></div>
<div class="row">
  <input id="q" type="text" placeholder="ask something about a Python package…"
     onkeydown="if(event.key==='Enter')send()">
  <button id="go" onclick="send()">Ask</button>
</div>
<div class="opts">
  <label><input type="checkbox" id="rag" checked> retrieval grounding</label>
  <label>temp <input id="t" type="range" min="10" max="130" value="70"
    oninput="tv.textContent=(this.value/100).toFixed(2)" style="width:110px"></label>
  <span id="tv">0.70</span>
</div>
<div class="opts" id="ex"></div>
</div><script>
const EX=["What is beautifulsoup4?","How do I install black?","What license does requests use?",
          "Which package should I use for progress bars?","How do I read a text file in Python?",
          "What is 128 + 46?"];
ex.innerHTML = EX.map(e=>`<span class="chip" onclick="q.value=this.textContent;send()">${e}</span>`).join('');
fetch('/info').then(r=>r.json()).then(d=>{meta.textContent =
  d.params.toLocaleString()+' params · '+d.stage+' checkpoint '+d.checkpoint+' (step '+d.step+
  ') · knowledge base '+d.kb.toLocaleString()+' packages';});
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;
  d.innerHTML=html;log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);}
async function send(){
  const text=q.value.trim(); if(!text) return; q.value='';
  add('you',escapeHtml(text)); go.disabled=true;
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:text,use_context:rag.checked,temperature:+t.value/100})});
  const d=await r.json();
  add('bot',escapeHtml(d.answer||'(empty)')+
    (d.context?'<div class="ctx">retrieved: '+escapeHtml(d.context.slice(0,240))+'…</div>':''));
  go.disabled=false; q.focus();
}
function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
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
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/info":
            maybe_reload()
            a = ASSISTANT
            return self._send(200, json.dumps({
                "params": a.model.num_params(), "step": a.step, "stage": a.stage,
                "checkpoint": os.path.basename(a.path), "kb": len(a.retriever.docs)}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path not in ("/chat", "/generate"):
            return self._send(404, json.dumps({"error": "not found"}))
        maybe_reload()
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        with LOCK, torch.no_grad():
            if self.path == "/chat":
                out = ASSISTANT.answer(req.get("message", ""),
                                       max_tokens=int(req.get("tokens", 140)),
                                       temperature=float(req.get("temperature", 0.7)),
                                       use_context=bool(req.get("use_context", True)))
            else:  # raw continuation, no chat template
                tok = ASSISTANT.tok
                ids = tok.encode(req.get("prompt", "")).ids or [ASSISTANT.eot]
                x = torch.tensor([ids[-ASSISTANT.model.cfg.block_size:]], dtype=torch.long)
                gen = ASSISTANT.model.generate(x, max_new_tokens=int(req.get("tokens", 160)),
                                               temperature=float(req.get("temperature", 0.8)),
                                               top_k=40, top_p=0.95)
                out = {"answer": tok.decode(gen[0].tolist(), skip_special_tokens=False)}
        self._send(200, json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    maybe_reload()
    print(f"[serve] http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
