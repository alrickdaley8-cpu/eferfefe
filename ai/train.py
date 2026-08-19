"""Pretrain the ~5M parameter LM on ~100M tokens — resumable, self-checkpointing, daemon-friendly.

Upgrades over the first version:
  * crash/interrupt safe      – SIGTERM/SIGINT save a checkpoint and exit 0, so a supervisor can
                                restart it and lose nothing
  * exact resume              – step, optimiser state and the data-sampling RNG are restored
  * best-checkpoint tracking  – keeps ai/checkpoints/best.pt at the lowest validation loss
  * heartbeat                 – ai/checkpoints/status.json is rewritten every log interval with
                                stage / step / tokens / loss / tok-per-s / ETA for `python -m ai.status`
  * mid-training data mix     – the last part of pretraining blends in instruction data
                                (--sft-mix-frac) so the base model already knows the chat format
  * throughput                – batches assembled with one vectorised gather; flush-to-zero denormals

Usage (normally started by ai/daemon.sh, which keeps it alive in the background):
    python -m ai.train --total-tokens 100000000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

from ai.model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
CKPT_DIR = os.path.join(ROOT, "ai", "checkpoints")
STATUS = os.path.join(CKPT_DIR, "status.json")

_stop = {"flag": False}


def _handle_signal(signum, frame):  # noqa: ARG001
    _stop["flag"] = True
    print(f"[train] signal {signum} received — will checkpoint and exit", flush=True)


def mark_done(stage: str) -> None:
    """Marker file the supervisor uses to know a stage never needs running again."""
    with open(os.path.join(CKPT_DIR, f"{stage}.done"), "w") as f:
        f.write(str(time.time()))


def write_status(**kw) -> None:
    kw["updated_at"] = time.time()
    kw["pid"] = os.getpid()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=2)
    os.replace(tmp, STATUS)


def get_batch(data: np.ndarray, bs: int, block: int, rng: np.random.Generator):
    """Vectorised window gather: one fancy-index instead of a Python loop."""
    ix = rng.integers(0, len(data) - block - 1, size=bs)
    off = np.arange(block + 1)
    win = data[(ix[:, None] + off[None, :])].astype(np.int64)
    return torch.from_numpy(win[:, :-1]), torch.from_numpy(win[:, 1:])


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step >= args.max_steps:
        return args.min_lr
    prog = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, data, args, iters: int = 20) -> float:
    model.eval()
    rng = np.random.default_rng(7)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, args.batch_size, args.block_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save(path: str, payload: dict) -> None:
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


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
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--ckpt-interval", type=int, default=250)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--time-budget", type=float, default=0.0, help="stop after N seconds (0 = off)")
    ap.add_argument("--sft-mix-frac", type=float, default=0.15,
                    help="fraction of batches drawn from instruction data during mid-training")
    ap.add_argument("--sft-start-frac", type=float, default=0.8,
                    help="training progress at which the instruction mix switches on")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(1337)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")

    sft_path = os.path.join(DATA, "qa", "sft_tokens.bin")
    sft_data = None
    if args.sft_mix_frac > 0 and os.path.exists(sft_path):
        sft_data = np.memmap(sft_path, dtype=np.uint16, mode="r")
        print(f"[train] mid-training mix: {args.sft_mix_frac:.0%} of batches from "
              f"{len(sft_data):,} instruction tokens after {args.sft_start_frac:.0%} progress",
              flush=True)
    elif args.sft_mix_frac > 0:
        print("[train] no instruction tokens found (run ai/build_qa.py + ai.finetune --repack) "
              "— continuing without the mid-training mix", flush=True)

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    args.max_steps = args.total_tokens // tokens_per_step

    cfg = GPTConfig(block_size=args.block_size)
    model = GPT(cfg)
    print(f"[train] params={model.num_params():,} tokens/step={tokens_per_step:,} "
          f"steps={args.max_steps:,} corpus={len(train_data):,} tokens", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, foreach=True)

    ckpt_path = os.path.join(CKPT_DIR, "ckpt.pt")
    start_step, best_val = 0, float("inf")
    rng = np.random.default_rng(1234)
    if args.resume == "auto" and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        if ck.get("rng_state") is not None:
            rng.bit_generator.state = ck["rng_state"]
        print(f"[train] resumed at step {start_step:,} ({start_step*tokens_per_step/1e6:.1f}M "
              f"tokens, best val {best_val:.4f})", flush=True)

    if start_step >= args.max_steps:
        print("[train] token budget already reached — nothing to do", flush=True)
        write_status(stage="pretrain", state="done", step=start_step,
                     tokens=start_step * tokens_per_step, total_tokens=args.total_tokens,
                     best_val=best_val)
        mark_done("pretrain")
        return

    logf = open(os.path.join(CKPT_DIR, "log.jsonl"), "a")
    t_start = t0 = time.time()
    ema = None
    step = start_step

    def checkpoint(step: int, val: float | None = None) -> None:
        payload = {"model": model.state_dict(), "optim": opt.state_dict(), "step": step,
                   "cfg": cfg.__dict__, "args": vars(args), "best_val": best_val,
                   "rng_state": rng.bit_generator.state}
        save(ckpt_path, payload)
        save(os.path.join(CKPT_DIR, "model.pt"),
             {"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "stage": "pretrain"})
        if val is not None and val <= best_val:
            save(os.path.join(CKPT_DIR, "best.pt"),
                 {"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                  "stage": "pretrain", "val_loss": val})

    for step in range(start_step, args.max_steps):
        for g in opt.param_groups:
            g["lr"] = lr = lr_at(step, args)

        mix = (sft_data is not None
               and step / args.max_steps >= args.sft_start_frac
               and rng.random() < args.sft_mix_frac)
        source = sft_data if mix else train_data

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(source, args.batch_size, args.block_size, rng)
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema = total_loss if ema is None else 0.9 * ema + 0.1 * total_loss

        if (step + 1) % args.log_interval == 0:
            dt = (time.time() - t0) / args.log_interval
            t0 = time.time()
            seen = (step + 1) * tokens_per_step
            tps = tokens_per_step / dt
            eta = (args.max_steps - step - 1) * dt
            rec = {"stage": "pretrain", "step": step + 1, "loss": round(total_loss, 4),
                   "ema": round(ema, 4), "lr": round(lr, 6), "tok": seen,
                   "tok_per_s": round(tps, 1), "eta_s": round(eta)}
            print(f"[train] step {step+1}/{args.max_steps} loss {total_loss:.4f} "
                  f"(ema {ema:.4f}) ppl {math.exp(min(ema,20)):.1f} lr {lr:.2e} {tps:.0f} tok/s "
                  f"{seen/1e6:.2f}M/{args.total_tokens/1e6:.0f}M  eta {eta/3600:.2f}h", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            write_status(stage="pretrain", state="running", step=step + 1,
                         total_steps=args.max_steps, tokens=seen, total_tokens=args.total_tokens,
                         loss=round(ema, 4), best_val=None if best_val == float("inf") else best_val,
                         tok_per_s=round(tps, 1), eta_s=round(eta),
                         elapsed_s=round(time.time() - t_start))

        val = None
        if (step + 1) % args.eval_interval == 0:
            val = evaluate(model, val_data, args)
            best_val = min(best_val, val)
            print(f"[train] *** step {step+1} val_loss {val:.4f} ppl {math.exp(min(val,20)):.2f} "
                  f"(best {best_val:.4f})", flush=True)
            logf.write(json.dumps({"stage": "pretrain", "step": step + 1,
                                   "val_loss": round(val, 4)}) + "\n")
            logf.flush()

        if val is not None or (step + 1) % args.ckpt_interval == 0 or step + 1 == args.max_steps:
            checkpoint(step + 1, val)

        if _stop["flag"] or (args.time_budget and time.time() - t_start > args.time_budget):
            checkpoint(step + 1)
            write_status(stage="pretrain", state="paused", step=step + 1,
                         total_steps=args.max_steps, tokens=(step + 1) * tokens_per_step,
                         total_tokens=args.total_tokens, loss=round(ema, 4))
            print(f"[train] stopped cleanly at step {step+1}", flush=True)
            sys.exit(0)

    write_status(stage="pretrain", state="done", step=args.max_steps,
                 total_steps=args.max_steps, tokens=args.max_steps * tokens_per_step,
                 total_tokens=args.total_tokens, loss=round(ema or 0.0, 4),
                 best_val=None if best_val == float("inf") else best_val)
    mark_done("pretrain")
    print("[train] pretraining complete", flush=True)


if __name__ == "__main__":
    main()
