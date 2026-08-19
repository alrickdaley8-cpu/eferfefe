"""Ask the tiny model questions: retrieval → prompt → generation, with a visible trace.

    python -m ai.chat "what is beautifulsoup4?"          # one shot
    python -m ai.chat --think "how do I install black?"  # show the reasoning trace
    python -m ai.chat --list                             # available checkpoints
    python -m ai.chat                                    # interactive REPL
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Iterator

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ai.finetune import format_prompt
from ai.model import GPT, GPTConfig
from ai.retrieve import Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
CKPT_DIR = os.path.join(ROOT, "ai", "checkpoints")

# nicer names for the checkpoints the pipeline produces
LABELS = {
    "sft.pt":       ("Chat (instruction tuned)", "the finished assistant"),
    "sft_demo.pt":  ("Chat preview", "early instruction tuning, trained while pretraining ran"),
    "model.pt":     ("Base (latest)", "current pretraining checkpoint, completion only"),
    "best.pt":      ("Base (best val)", "pretraining checkpoint with the lowest validation loss"),
    "base_demo.pt": ("Base snapshot", "frozen early pretraining checkpoint"),
}
HIDDEN = {"ckpt.pt"}          # optimiser-state checkpoint, not a servable model


def list_checkpoints() -> list[dict]:
    out = []
    for fn in sorted(os.listdir(CKPT_DIR)):
        if not fn.endswith(".pt") or fn in HIDDEN:
            continue
        path = os.path.join(CKPT_DIR, fn)
        label, desc = LABELS.get(fn, (fn, ""))
        info = {"file": fn, "label": label, "description": desc,
                "size_mb": round(os.path.getsize(path) / 1e6, 1),
                "modified": os.path.getmtime(path)}
        try:
            meta = torch.load(path, map_location="meta", weights_only=False, mmap=True)
            info["stage"] = meta.get("stage", "pretrain")
            info["step"] = meta.get("step", 0)
            info["val_loss"] = meta.get("val_loss")
        except Exception:
            info["stage"], info["step"] = "?", 0
        info["tokens"] = info["step"] * 8192
        out.append(info)
    out.sort(key=lambda d: (d["stage"] != "sft", -d["modified"]))
    return out


def default_ckpt() -> str:
    for name in ("sft.pt", "sft_demo.pt", "best.pt", "model.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no checkpoint in ai/checkpoints/")


_RETRIEVER: Retriever | None = None


def shared_retriever() -> Retriever:
    """The knowledge base is ~9 MB of JSON; build it once and share it across models."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


class Assistant:
    def __init__(self, ckpt: str | None = None, threads: int | None = None):
        torch.set_num_threads(threads or (os.cpu_count() or 2))
        self.path = ckpt or default_ckpt()
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig(**ck["cfg"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.step = ck.get("step", 0)
        self.stage = ck.get("stage", "pretrain")
        self.tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
        self.eot = self.tok.token_to_id("<|endoftext|>")
        self.retriever = shared_retriever()

    # ------------------------------------------------------------------ prompting
    def build_prompt(self, question: str, use_context: bool = True,
                     chat_template: bool = True) -> tuple[str, str | None, list[dict]]:
        candidates: list[dict] = []
        ctx = None
        if use_context:
            hits = self.retriever.search(question, k=3)
            for score, doc in hits:
                candidates.append({"name": doc["name"], "score": round(score, 2),
                                   "summary": doc.get("summary", "")[:110]})
            if hits and hits[0][0] >= 1.5:
                ctx = self.retriever.context_of(hits[0][1])
        q = f"Context: {ctx}\nQuestion: {question}" if ctx else question
        prompt = format_prompt(q) if chat_template else q
        return prompt, ctx, candidates

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def stream(self, question: str, max_tokens: int = 160, temperature: float = 0.7,
               top_k: int = 40, top_p: float = 0.9, use_context: bool = True,
               chat_template: bool = True) -> Iterator[dict]:
        """Yield the whole thought process: retrieval → prompt → token-by-token decoding."""
        t0 = time.time()
        yield {"type": "thought", "kind": "retrieve",
               "text": f"searching {len(self.retriever.docs):,} package records"
                       if use_context else "retrieval disabled — answering closed-book"}

        prompt, ctx, cands = self.build_prompt(question, use_context, chat_template)
        if use_context:
            if cands:
                yield {"type": "thought", "kind": "candidates", "candidates": cands}
            if ctx:
                yield {"type": "thought", "kind": "context", "text": ctx}
            else:
                yield {"type": "thought", "kind": "context",
                       "text": "no confident match — answering without context"}

        ids = self.tok.encode(prompt).ids
        keep = self.model.cfg.block_size - max_tokens
        if len(ids) > keep:
            ids = ids[:1] + ids[-(keep - 1):]
            yield {"type": "thought", "kind": "trim",
                   "text": f"prompt trimmed to the last {keep} tokens (512-token window)"}
        yield {"type": "thought", "kind": "prompt", "text": prompt,
               "tokens": len(ids), "budget": max_tokens}

        x = torch.tensor([ids], dtype=torch.long)
        out_ids: list[int] = []
        t_gen = time.time()
        for i in range(max_tokens):
            logits, _ = self.model(x[:, -self.model.cfg.block_size:])
            logits = logits[0, -1, :] / max(temperature, 1e-5)
            probs_full = F.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs_full, 5)
            filt = logits.clone()
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                filt[filt < kth] = -float("inf")
            probs = F.softmax(filt, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(0)
                sp[cum - sp > top_p] = 0.0
                sp /= sp.sum()
                nxt = si[torch.multinomial(sp, 1)]
            else:
                nxt = torch.multinomial(probs, 1)
            tid = int(nxt)
            if tid == self.eot:
                break
            out_ids.append(tid)
            x = torch.cat([x, nxt.view(1, 1)], dim=1)
            piece = self.tok.decode([tid], skip_special_tokens=False)
            yield {"type": "token", "text": piece, "i": i,
                   "confidence": round(float(probs_full[tid]), 3),
                   "alternatives": [{"token": self.tok.decode([int(t)], skip_special_tokens=False),
                                     "p": round(float(p), 3)} for p, t in zip(topv, topi)]}
            text_so_far = self.tok.decode(out_ids, skip_special_tokens=False)
            if "### Question" in text_so_far or "<|endoftext|>" in text_so_far:
                break

        text = self.tok.decode(out_ids, skip_special_tokens=False)
        text = text.split("<|endoftext|>")[0].split("### Question")[0].strip()
        dt = time.time() - t_gen
        yield {"type": "done", "answer": text, "context": ctx,
               "stats": {"prompt_tokens": len(ids), "generated_tokens": len(out_ids),
                         "tok_per_s": round(len(out_ids) / dt, 1) if dt else 0,
                         "total_s": round(time.time() - t0, 2)},
               "checkpoint": os.path.basename(self.path), "stage": self.stage, "step": self.step}

    def answer(self, question: str, **kw) -> dict:
        result = {"answer": "", "context": None}
        for ev in self.stream(question, **kw):
            if ev["type"] == "done":
                result = ev
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--list", action="store_true", help="show available checkpoints")
    ap.add_argument("--think", action="store_true", help="print the retrieval/decoding trace")
    ap.add_argument("--tokens", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in list_checkpoints():
            print(f"{c['file']:<14} {c['label']:<28} stage={c['stage']:<8} step={c['step']:>6,} "
                  f"({c['tokens']/1e6:.1f}M tokens)")
        return

    a = Assistant(args.ckpt)
    print(f"# {a.stage} checkpoint {os.path.basename(a.path)} (step {a.step:,}), "
          f"{a.model.num_params():,} params, {len(a.retriever.docs):,} KB docs\n")

    def ask(q: str) -> None:
        for ev in a.stream(q, max_tokens=args.tokens, temperature=args.temperature,
                           use_context=not args.no_context):
            if ev["type"] == "thought" and args.think:
                if ev["kind"] == "candidates":
                    for c in ev["candidates"]:
                        print(f"  · {c['score']:>7.2f}  {c['name']}: {c['summary']}")
                else:
                    print(f"  [{ev['kind']}] {ev['text'][:160]}")
            elif ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                s = ev["stats"]
                print(f"\n  ({s['generated_tokens']} tokens, {s['tok_per_s']} tok/s)\n")

    if args.question:
        ask(" ".join(args.question))
        return
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q:
            ask(q)


if __name__ == "__main__":
    main()
