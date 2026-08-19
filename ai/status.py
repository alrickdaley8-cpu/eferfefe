"""Print the state of the background training pipeline.

    python -m ai.status          # snapshot
    python -m ai.status --watch  # refresh every 30s
"""
from __future__ import annotations

import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "ai", "checkpoints")


def human(sec: float) -> str:
    sec = int(max(0, sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


def bar(frac: float, width: int = 34) -> str:
    n = int(max(0.0, min(1.0, frac)) * width)
    return "[" + "#" * n + "-" * (width - n) + "]"


def load(name: str) -> dict | None:
    p = os.path.join(CKPT, name)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def tail_val(path: str, key: str = "val_loss") -> float | None:
    p = os.path.join(CKPT, path)
    if not os.path.exists(p):
        return None
    last = None
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if key in r:
                last = r[key]
    return last


def show() -> None:
    s = load("status.json")
    if not s:
        print("no status yet — start the daemon with ai/daemon.sh start")
        return
    stage = s.get("stage", "?")
    tok, total = s.get("tokens", 0), s.get("total_tokens", 1)
    frac = tok / max(1, total)
    age = time.time() - s.get("updated_at", 0)
    alive = "live" if age < 300 else f"stale ({human(age)} old)"

    print(f"stage      : {stage}  ({s.get('state', '?')}, {alive}, pid {s.get('pid')})")
    print(f"progress   : {bar(frac)} {frac*100:5.1f}%   {tok/1e6:.2f}M / {total/1e6:.0f}M tokens")
    print(f"step       : {s.get('step', 0):,} / {s.get('total_steps', 0):,}")
    if s.get("loss") is not None:
        print(f"train loss : {s['loss']}")
    vl = tail_val("log.jsonl" if stage == "pretrain" else "sft_log.jsonl")
    if vl is not None:
        print(f"val loss   : {vl}")
    if s.get("best_val") is not None:
        print(f"best val   : {s['best_val']}")
    if s.get("tok_per_s"):
        print(f"throughput : {s['tok_per_s']:.0f} tok/s")
    if s.get("eta_s"):
        print(f"eta        : {human(s['eta_s'])} for this stage")
    ck = [f for f in ("ckpt.pt", "model.pt", "best.pt", "sft.pt") 
          if os.path.exists(os.path.join(CKPT, f))]
    print(f"checkpoints: {', '.join(ck) or 'none yet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        show()
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
