"""Train the ~5M parameter LM on ~100M tokens.

    python ai/train.py --total-tokens 100_000_000

The run is fully resumable: it checkpoints every --ckpt-interval steps to
ai/checkpoints/ckpt.pt and picks up from there automatically (--resume auto).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from ai.model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
CKPT_DIR = os.path.join(ROOT, "ai", "checkpoints")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i: i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1: i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step >= args.max_steps:
        return args.min_lr
    prog = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, data, args, rng, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, args.batch_size, args.block_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--ckpt-interval", type=int, default=250)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--time-budget", type=float, default=0.0, help="stop after N seconds (0 = no limit)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    args.max_steps = args.total_tokens // tokens_per_step

    cfg = GPTConfig(block_size=args.block_size)
    model = GPT(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[train] device={device} params={model.num_params():,} "
          f"tokens/step={tokens_per_step:,} steps={args.max_steps:,} "
          f"train_tokens_available={len(train_data):,}", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=False,
    )

    start_step = 0
    ckpt_path = os.path.join(CKPT_DIR, "ckpt.pt")
    if args.resume == "auto" and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step = ck["step"]
        print(f"[train] resumed from step {start_step}", flush=True)

    rng = np.random.default_rng(1234 + start_step)
    log_path = os.path.join(CKPT_DIR, "log.jsonl")
    logf = open(log_path, "a")
    t_start = time.time()
    t0 = time.time()
    running = None

    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.block_size, rng)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        running = total_loss if running is None else 0.9 * running + 0.1 * total_loss

        if (step + 1) % args.log_interval == 0:
            dt = (time.time() - t0) / args.log_interval
            t0 = time.time()
            seen = (step + 1) * tokens_per_step
            rec = {"step": step + 1, "loss": round(total_loss, 4), "ema": round(running, 4),
                   "lr": round(lr, 6), "tok": seen, "tok_per_s": round(tokens_per_step / dt, 1),
                   "s_per_step": round(dt, 3), "elapsed": round(time.time() - t_start, 1)}
            print(f"[train] step {step+1}/{args.max_steps} loss {total_loss:.4f} "
                  f"(ema {running:.4f}) ppl {math.exp(min(running,20)):.1f} lr {lr:.2e} "
                  f"{rec['tok_per_s']:.0f} tok/s  {seen/1e6:.2f}M/{args.total_tokens/1e6:.0f}M tokens",
                  flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        if (step + 1) % args.eval_interval == 0:
            vl = evaluate(model, val_data, args, np.random.default_rng(7))
            print(f"[train] *** step {step+1} val_loss {vl:.4f} val_ppl {math.exp(min(vl,20)):.2f}",
                  flush=True)
            logf.write(json.dumps({"step": step + 1, "val_loss": round(vl, 4)}) + "\n")
            logf.flush()

        if (step + 1) % args.ckpt_interval == 0 or step + 1 == args.max_steps:
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                        "step": step + 1, "cfg": cfg.__dict__, "args": vars(args)},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step + 1},
                       os.path.join(CKPT_DIR, "model.pt"))

        if args.time_budget and time.time() - t_start > args.time_budget:
            print(f"[train] time budget reached at step {step+1}", flush=True)
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                        "step": step + 1, "cfg": cfg.__dict__, "args": vars(args)}, ckpt_path)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step + 1},
                       os.path.join(CKPT_DIR, "model.pt"))
            break

    print("[train] finished", flush=True)


if __name__ == "__main__":
    main()
