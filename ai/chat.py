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
import re
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
# questions that are self-contained: attaching a package blurb only distracts the model
SELF_CONTAINED = re.compile(
    r"\d+\s*[-+x*/]\s*\d+|\bhow many (letters|vowels|times)\b|\breverse the\b|"
    r"\baverage of\b|\bwhat comes next\b|\b\d+% of\b|\bwhich is bigger\b|"
    r"\btimes\b.*\d|\bfirst and last letters\b", re.I)
MIN_CONTEXT_SCORE = 8.0        # BM25 score below which a match is not trustworthy

LABELS = {
    "sft.pt":       ("Chat (instruction tuned)", "the finished assistant"),
    "sft_demo.pt":  ("Chat preview", "early instruction tuning, trained while pretraining ran"),
    "model.pt":     ("Base (latest)", "current pretraining checkpoint, completion only"),
    "best.pt":      ("Base (best val)", "pretraining checkpoint with the lowest validation loss"),
    "base_demo.pt": ("Base snapshot", "frozen early pretraining checkpoint"),
}
HIDDEN = {"ckpt.pt"}          # optimiser-state checkpoints are not servable models


def _servable(fn: str) -> bool:
    return fn.endswith(".pt") and fn not in HIDDEN and not fn.endswith("_ckpt.pt")


def list_checkpoints() -> list[dict]:
    out = []
    for fn in sorted(os.listdir(CKPT_DIR)):
        if not _servable(fn):
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
    def __init__(self, ckpt: str | None = None, threads: int | None = None,
                 quantize: bool | None = None):
        torch.set_num_threads(threads or int(os.environ.get("LM_THREADS", os.cpu_count() or 2)))
        self.path = ckpt or default_ckpt()
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig(**ck["cfg"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        if quantize is None:
            quantize = os.environ.get("LM_QUANTIZE", "0") == "1"
        self.quantized = False
        if quantize:
            # dynamic int8 on the linear layers: ~1.5-2x faster matmuls on CPU
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.quantized = True
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
        if use_context and SELF_CONTAINED.search(question):
            use_context = False
            self._skip_reason = "arithmetic/string task — answering without retrieval"
        else:
            self._skip_reason = None
        if use_context:
            hits = self.retriever.search(question, k=3)
            for score, doc in hits:
                candidates.append({"name": doc["name"], "score": round(score, 2),
                                   "summary": doc.get("summary", "")[:110]})
            if hits and hits[0][0] >= MIN_CONTEXT_SCORE:
                # comparison questions need both packages in the context, numbered exactly the
                # way the multi-hop training examples were written
                named = [h for h in hits[:2]
                         if h[1]["name"].lower() in question.lower()
                         and h[0] >= MIN_CONTEXT_SCORE]
                if len(named) == 2:
                    ctx = (f"(1) {self.retriever.context_of(named[0][1])}\n"
                           f"(2) {self.retriever.context_of(named[1][1])}")
                else:
                    ctx = self.retriever.context_of(hits[0][1])
        q = f"Context: {ctx}\nQuestion: {question}" if ctx else question
        prompt = format_prompt(q) if chat_template else q
        return prompt, ctx, candidates

    # ------------------------------------------------------------------ generation
    @torch.inference_mode()
    def stream(self, question: str, max_tokens: int = 160, temperature: float = 0.7,
               top_k: int = 40, top_p: float = 0.9, use_context: bool = True,
               chat_template: bool = True) -> Iterator[dict]:
        """Yield the whole thought process: retrieval → prompt → token-by-token decoding."""
        t0 = time.time()
        prompt, ctx, cands = self.build_prompt(question, use_context, chat_template)
        skipped = getattr(self, "_skip_reason", None)
        yield {"type": "thought", "kind": "retrieve",
               "text": skipped or (f"searching {len(self.retriever.docs):,} package records"
                                   if use_context else "retrieval disabled — answering closed-book")}
        use_context = use_context and skipped is None
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
        cache = None                # KV cache: the prompt is encoded once, then one token a step
        out_ids: list[int] = []
        channel = "answer"          # flips to "think" inside a <think> … </think> block
        pending = ""
        tail = ""                   # last few characters, for stop-sequence checks
        t_gen = time.time()
        for i in range(max_tokens):
            if cache is None:
                logits, _, cache = self.model(x, use_cache=True, last_only=True)
            else:
                logits, _, cache = self.model(x[:, -1:], past=cache, use_cache=True,
                                              last_only=True)
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
            if x.size(1) >= self.model.cfg.block_size:
                break                                   # 512-token window is full
            alts = [{"token": self.tok.decode([int(t)], skip_special_tokens=False),
                     "p": round(float(p), 3)} for p, t in zip(topv, topi)]
            piece = self.tok.decode([tid], skip_special_tokens=False)
            pending += piece
            tail = (tail + piece)[-24:]

            # Split the stream into a reasoning channel and an answer channel, without ever
            # leaking a partially decoded "<think>" tag to the client.
            while True:
                tag = "</think>" if channel == "think" else "<think>"
                cut = pending.find(tag)
                if cut < 0:
                    break
                head, pending = pending[:cut], pending[cut + len(tag):]
                if head:
                    yield {"type": "token", "text": head, "channel": channel, "i": i,
                           "confidence": round(float(probs_full[tid]), 3),
                           "alternatives": alts}
                channel = "answer" if channel == "think" else "think"

            safe = pending
            hold = 0
            for cand in ("<think>", "</think>"):
                for L in range(1, min(len(cand), len(pending)) + 1):
                    if pending.endswith(cand[:L]):
                        hold = max(hold, L)
            if hold:
                safe, pending = pending[:-hold], pending[-hold:]
            else:
                pending = ""
            if safe:
                yield {"type": "token", "text": safe, "channel": channel, "i": i,
                       "confidence": round(float(probs_full[tid]), 3),
                       "alternatives": alts}
            if "### Question" in tail or "<|endoftext|>" in tail:
                break

        raw = self.tok.decode(out_ids, skip_special_tokens=False)
        raw = raw.split("<|endoftext|>")[0].split("### Question")[0].strip()
        reasoning = ""
        if "<think>" in raw:
            head, _, rest = raw.partition("<think>")
            reasoning, _, tail = rest.partition("</think>")
            text = (head + tail).strip()
            reasoning = reasoning.strip()
        else:
            text = raw
        if not text and reasoning:
            # the model never closed its <think> block: fall back to its last line
            text = reasoning.strip().splitlines()[-1]
        dt = time.time() - t_gen
        yield {"type": "done", "answer": text, "reasoning": reasoning, "raw": raw, "context": ctx,
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
                if ev.get("channel") == "think" and not args.think:
                    continue
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
