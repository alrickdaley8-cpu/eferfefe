"""Instruction / QA fine-tuning (SFT) of the pretrained 5M model.

Chat format (plain text, so the tokenizer needs no new symbols):

    ### Question:
    {prompt}

    ### Answer:
    {response}<|endoftext|>

Loss is computed on the answer tokens only. Examples are packed into a flat token stream with a
parallel loss mask, and sampled in 512-token windows exactly like pretraining.

    python -m ai.finetune --tokens 15000000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from ai.model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
QA_DIR = os.path.join(DATA, "qa")
CKPT_DIR = os.path.join(ROOT, "ai", "checkpoints")

Q_HEAD, A_HEAD = "### Question:\n", "\n\n### Answer:\n"
IGNORE = -100


def format_prompt(question: str) -> str:
    return f"{Q_HEAD}{question}{A_HEAD}"


def pack(tokenizer: Tokenizer, path: str, tok_bin: str, mask_bin: str, max_len: int = 480) -> None:
    """Tokenize sft.jsonl into a flat uint16 stream + uint8 answer mask."""
    eot = tokenizer.token_to_id("<|endoftext|>")
    rows = [json.loads(l) for l in open(path)]
    toks, masks = [], []
    B = 2000
    with open(tok_bin, "wb") as ft, open(mask_bin, "wb") as fm:
        for i in range(0, len(rows), B):
            chunk = rows[i: i + B]
            pe = tokenizer.encode_batch([format_prompt(r["prompt"]) for r in chunk])
            re_ = tokenizer.encode_batch([r["response"] for r in chunk])
            for p, a in zip(pe, re_):
                ids = p.ids + a.ids + [eot]
                if len(ids) > max_len:
                    continue
                m = [0] * len(p.ids) + [1] * (len(a.ids) + 1)
                toks.extend(ids)
                masks.extend(m)
            np.asarray(toks, dtype=np.uint16).tofile(ft)
            np.asarray(masks, dtype=np.uint8).tofile(fm)
            toks, masks = [], []
            if (i // B) % 10 == 0:
                print(f"[sft] packed {i + len(chunk):,}/{len(rows):,} examples", flush=True)


def get_batch(tokens, mask, bs, block, rng):
    ix = rng.integers(0, len(tokens) - block - 1, size=bs)
    x = np.stack([tokens[i: i + block] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1: i + 1 + block] for i in ix]).astype(np.int64)
    m = np.stack([mask[i + 1: i + 1 + block] for i in ix]).astype(bool)
    y = np.where(m, y, IGNORE)
    return torch.from_numpy(x), torch.from_numpy(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=15_000_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--ckpt-interval", type=int, default=200)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--base", default=os.path.join(CKPT_DIR, "model.pt"))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "sft.pt"))
    ap.add_argument("--repack", action="store_true")
    ap.add_argument("--time-budget", type=float, default=0.0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tokenizer = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))

    tok_bin = os.path.join(QA_DIR, "sft_tokens.bin")
    mask_bin = os.path.join(QA_DIR, "sft_mask.bin")
    if args.repack or not os.path.exists(tok_bin):
        pack(tokenizer, os.path.join(QA_DIR, "sft.jsonl"), tok_bin, mask_bin)

    tokens = np.memmap(tok_bin, dtype=np.uint16, mode="r")
    mask = np.memmap(mask_bin, dtype=np.uint8, mode="r")
    n_val = min(200_000, len(tokens) // 20)
    tr_tok, tr_mask = tokens[:-n_val], mask[:-n_val]
    va_tok, va_mask = tokens[-n_val:], mask[-n_val:]

    ck = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.train()

    tps = args.batch_size * args.block_size
    steps = args.tokens // tps
    print(f"[sft] base step {ck.get('step')} | {model.num_params():,} params | "
          f"{len(tokens):,} SFT tokens ({mask[:].mean()*100:.1f}% answer tokens) | "
          f"{steps:,} steps x {tps:,} tok", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    rng = np.random.default_rng(7)
    logf = open(os.path.join(CKPT_DIR, "sft_log.jsonl"), "a")
    t_start = t0 = time.time()
    ema = None
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(tr_tok, tr_mask, args.batch_size, args.block_size, rng)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()

        if (step + 1) % args.log_interval == 0:
            dt = (time.time() - t0) / args.log_interval
            t0 = time.time()
            rec = {"step": step + 1, "loss": round(loss.item(), 4), "ema": round(ema, 4),
                   "tok": (step + 1) * tps, "tok_per_s": round(tps / dt, 1)}
            print(f"[sft] step {step+1}/{steps} loss {loss.item():.4f} (ema {ema:.4f}) "
                  f"ppl {math.exp(min(ema,20)):.1f} {rec['tok_per_s']:.0f} tok/s "
                  f"{(step+1)*tps/1e6:.2f}M/{args.tokens/1e6:.0f}M", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        if (step + 1) % args.ckpt_interval == 0 or step + 1 == steps:
            model.eval()
            with torch.no_grad():
                vs = []
                vrng = np.random.default_rng(3)
                for _ in range(10):
                    vx, vy = get_batch(va_tok, va_mask, args.batch_size, args.block_size, vrng)
                    vl, _ = model(vx)
                    vs.append(torch.nn.functional.cross_entropy(
                        vl.view(-1, vl.size(-1)), vy.reshape(-1), ignore_index=IGNORE).item())
            model.train()
            v = float(np.mean(vs))
            print(f"[sft] *** step {step+1} val_loss {v:.4f} ppl {math.exp(min(v,20)):.2f}", flush=True)
            logf.write(json.dumps({"step": step + 1, "val_loss": round(v, 4)}) + "\n")
            logf.flush()
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "step": step + 1, "stage": "sft"}, args.out + ".tmp")
            os.replace(args.out + ".tmp", args.out)

        if args.time_budget and time.time() - t_start > args.time_budget:
            print("[sft] time budget reached", flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "step": step + 1, "stage": "sft"}, args.out)
            break
    print("[sft] finished", flush=True)


if __name__ == "__main__":
    main()
