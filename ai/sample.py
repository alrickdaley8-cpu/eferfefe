"""Generate text from a trained checkpoint.

    python -m ai.sample --prompt "# file: README.md" --tokens 200
"""
from __future__ import annotations

import argparse
import os

import torch
from tokenizers import Tokenizer

from ai.model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
CKPT = os.path.join(ROOT, "ai", "checkpoints", "model.pt")


def load(ckpt_path: str = CKPT):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    return model, tok, ck.get("step", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="# file: README.md")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model, tok, step = load(args.ckpt)
    eot = tok.token_to_id("<|endoftext|>")
    ids = tok.encode(args.prompt).ids or [eot]
    x = torch.tensor([ids], dtype=torch.long)
    print(f"# checkpoint step {step}, {model.num_params():,} params\n")
    for i in range(args.samples):
        out = model.generate(x, max_new_tokens=args.tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, eot_id=eot)
        text = tok.decode(out[0].tolist(), skip_special_tokens=False)
        print(f"----- sample {i+1} -----\n{text}\n")


if __name__ == "__main__":
    main()
