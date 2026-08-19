"""Ask the fine-tuned tiny model questions (retrieval-grounded).

    python -m ai.chat "what is beautifulsoup4?"
    python -m ai.chat            # interactive REPL
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from tokenizers import Tokenizer

from ai.finetune import format_prompt
from ai.model import GPT, GPTConfig
from ai.retrieve import Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
CKPT_DIR = os.path.join(ROOT, "ai", "checkpoints")


def default_ckpt() -> str:
    for name in ("sft.pt", "sft_demo.pt", "model.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no checkpoint in ai/checkpoints/")


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
        self.retriever = Retriever()

    def build_prompt(self, question: str, use_context: bool = True) -> tuple[str, str | None]:
        ctx = self.retriever.context_for(question) if use_context else None
        q = f"Context: {ctx}\nQuestion: {question}" if ctx else question
        return format_prompt(q), ctx

    @torch.no_grad()
    def answer(self, question: str, max_tokens: int = 140, temperature: float = 0.7,
               top_k: int = 40, top_p: float = 0.9, use_context: bool = True) -> dict:
        prompt, ctx = self.build_prompt(question, use_context)
        ids = self.tok.encode(prompt).ids[-self.model.cfg.block_size + max_tokens:]
        x = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                                  top_k=top_k, top_p=top_p, eot_id=self.eot)
        text = self.tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=False)
        text = text.split("<|endoftext|>")[0].split("### Question")[0].strip()
        return {"answer": text, "context": ctx, "prompt": prompt,
                "checkpoint": os.path.basename(self.path), "stage": self.stage, "step": self.step}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokens", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-context", action="store_true")
    args = ap.parse_args()

    a = Assistant(args.ckpt)
    print(f"# {a.stage} checkpoint {os.path.basename(a.path)} (step {a.step}), "
          f"{a.model.num_params():,} params, {len(a.retriever.docs):,} KB docs\n", file=sys.stderr)

    def ask(q: str) -> None:
        r = a.answer(q, max_tokens=args.tokens, temperature=args.temperature,
                     use_context=not args.no_context)
        if r["context"]:
            print(f"[retrieved] {r['context'][:120]}…")
        print(r["answer"] + "\n")

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
