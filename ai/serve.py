"""Web UI for the tiny LM: streaming chat, a visible thought process, and model switching.

    python -m ai.serve --port 8000

Endpoints
    GET  /                  the UI
    GET  /models            available checkpoints (+ which one is loaded)
    GET  /status            live training progress
    POST /chat/stream       server-sent events: thought → token → done
    POST /chat              non-streaming convenience wrapper
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ai.chat import CKPT_DIR, Assistant, default_ckpt, list_checkpoints
from ai.make_page import page_data
from ai.ui import PAGE

NOT_READY = ("The model has no checkpoint yet — training writes the first one at step 250. "
             "The page, retrieval and the grounded answer layer are already live.")

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")


def landing_page() -> str:
    """The generated project page (with the chat widget) if it exists, else the bare app."""
    try:
        with open(INDEX, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return PAGE

LOCK = threading.Lock()
_cache: "OrderedDict[str, tuple[float, Assistant]]" = OrderedDict()
_current = {"file": None}
MAX_CACHED = 2


def get_assistant(file: str | None = None) -> Assistant | None:
    """Load (and cache) a checkpoint by name; None while training has not written one yet."""
    try:
        file = file or _current["file"] or os.path.basename(default_ckpt())
        path = os.path.join(CKPT_DIR, file)
        if not os.path.exists(path):
            path = default_ckpt()
            file = os.path.basename(path)
    except FileNotFoundError:
        return None
    mtime = os.path.getmtime(path)
    with LOCK:
        hit = _cache.get(file)
        if hit and hit[0] >= mtime:
            _cache.move_to_end(file)
            _current["file"] = file
            return hit[1]
        a = Assistant(path)
        _cache[file] = (mtime, a)
        _cache.move_to_end(file)
        while len(_cache) > MAX_CACHED:
            _cache.popitem(last=False)
        _current["file"] = file
        print(f"[serve] loaded {file} ({a.stage}, step {a.step:,})", flush=True)
        return a


def training_status() -> dict:
    try:
        with open(os.path.join(CKPT_DIR, "status.json")) as f:
            s = json.load(f)
        s["live"] = (time.time() - s.get("updated_at", 0)) < 300
        return s
    except Exception:
        return {"state": "unknown", "live": False}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------------ routes
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, landing_page(), "text/html; charset=utf-8")
        if self.path == "/app":            # the chat-only view
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/models":
            models = list_checkpoints()
            a = get_assistant(_current["file"])
            if a is None:                       # server is up, model is not trained yet
                return self._send(200, json.dumps({
                    "models": [], "current": None, "ready": False, "params": 5_015_808,
                    "kb": 0, "vocab": 8192, "block_size": 512,
                    "message": "no checkpoint yet — the trainer writes its first one at step 250"}))
            cur = os.path.basename(a.path)
            for m in models:
                m["loaded"] = m["file"] == cur
            return self._send(200, json.dumps({
                "models": models, "current": cur, "ready": True, "params": a.model.num_params(),
                "kb": len(a.retriever.docs), "vocab": a.model.cfg.vocab_size,
                "block_size": a.model.cfg.block_size}))
        if self.path == "/status":
            return self._send(200, json.dumps(training_status()))
        if self.path == "/page-data":
            # everything the landing page shows, live — so a served page is never a stale snapshot
            try:
                return self._send(200, json.dumps(page_data()))
            except Exception as exc:
                return self._send(500, json.dumps({"error": str(exc)}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/chat":
            req = self._body()
            a = get_assistant(req.get("model"))
            if a is None:
                return self._send(503, json.dumps({"error": "model not ready",
                                                   "answer": NOT_READY}))
            with LOCK:
                out = a.answer(req.get("message", ""),
                               max_tokens=int(req.get("tokens", 160)),
                               temperature=float(req.get("temperature", 0.7)),
                               top_k=int(req.get("top_k", 40)),
                               use_context=bool(req.get("use_context", True)),
                               chat_template=bool(req.get("chat_template", True)),
                               grounded=bool(req.get("grounded", True)))
            return self._send(200, json.dumps(out))

        if self.path == "/chat/stream":
            req = self._body()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(ev: dict) -> None:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

            try:
                a = get_assistant(req.get("model"))
                if a is None:
                    emit({"type": "error", "text": NOT_READY})
                    emit({"type": "done", "answer": NOT_READY, "reasoning": "",
                          "verification": "unchecked", "context": None,
                          "stats": {"prompt_tokens": 0, "generated_tokens": 0,
                                    "tok_per_s": 0, "total_s": 0}})
                    return
                emit({"type": "thought", "kind": "model",
                      "text": f"{os.path.basename(a.path)} · {a.stage} checkpoint · "
                              f"step {a.step:,} ({a.step*8192/1e6:.1f}M tokens) · "
                              f"{a.model.num_params():,} params"})
                with LOCK:
                    for ev in a.stream(req.get("message", ""),
                                       max_tokens=int(req.get("tokens", 160)),
                                       temperature=float(req.get("temperature", 0.7)),
                                       top_k=int(req.get("top_k", 40)),
                                       use_context=bool(req.get("use_context", True)),
                                       chat_template=bool(req.get("chat_template", True)),
                                       grounded=bool(req.get("grounded", True))):
                        emit(ev)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # surface errors in the UI instead of hanging
                try:
                    emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass
            return

        self._send(404, json.dumps({"error": "not found"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    # one thread is measurably fastest for single-stream decoding of a model this small
    os.environ.setdefault("LM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    if get_assistant() is None:
        print("[serve] no checkpoint yet — serving the site; chat unlocks when one appears",
              flush=True)
    print(f"[serve] http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
