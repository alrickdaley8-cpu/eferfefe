"""Measure how often the assistant actually answers correctly.

Builds questions whose answers are known from the knowledge base (and from arithmetic), asks the
assistant twice — raw model output vs. the grounded/verified pipeline — and scores both.

    python -m ai.eval --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

from ai.chat import Assistant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "ai", "data", "qa", "knowledge.jsonl")


def build_cases(n: int, seed: int = 5) -> list[dict]:
    rng = random.Random(seed)
    docs = [json.loads(l) for l in open(KB)]
    rng.shuffle(docs)
    cases: list[dict] = []
    for d in docs:
        if len(cases) >= n * 0.75:
            break
        name = d["name"]
        if d.get("license"):
            cases.append({"q": f"What license does {name} use?", "expect": d["license"]})
        if d.get("version") and len(cases) < n:
            cases.append({"q": f"What is the latest version of {name}?", "expect": d["version"]})
        if d.get("deps") and len(cases) < n:
            dep = rng.choice(d["deps"])
            cases.append({"q": f"Does {name} depend on {dep}?", "expect": "yes"})
        if d.get("requires_python") and len(cases) < n:
            cases.append({"q": f"Which Python version does {name} need?",
                          "expect": d["requires_python"]})
    while len(cases) < n:                                    # arithmetic / string tail
        a, b = rng.randint(100, 9999), rng.randint(100, 9999)
        kind = rng.choice(["add", "mul", "letters"])
        if kind == "add":
            cases.append({"q": f"What is {a} + {b}?", "expect": str(a + b)})
        elif kind == "mul":
            x, y = rng.randint(2, 60), rng.randint(2, 60)
            cases.append({"q": f"What is {x} times {y}?", "expect": str(x * y)})
        else:
            w = rng.choice(["package", "installer", "template", "notebook", "compiler"])
            cases.append({"q": f"How many letters are in the word {w}?", "expect": str(len(w))})
    return cases[:n]


def scored(answer: str, expect: str) -> bool:
    a = answer.lower()
    e = expect.lower()
    if e == "yes":
        return a.startswith("yes") or ", yes" in a
    e = re.sub(r"\s+", " ", e).strip()
    return e in re.sub(r"\s+", " ", a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokens", type=int, default=70)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = build_cases(args.n)
    a = Assistant(args.ckpt)
    print(f"# {os.path.basename(a.path)} ({a.stage}, step {a.step:,}) on {len(cases)} questions\n")

    results = {}
    for mode, grounded in (("model only", False), ("grounded + verified", True)):
        hits, t0 = 0, time.time()
        for c in cases:
            r = a.answer(c["q"], max_tokens=args.tokens, temperature=0.2, top_k=5,
                         grounded=grounded)
            ok = scored(r["answer"], c["expect"])
            hits += ok
            if args.verbose and not ok:
                print(f"  ✗ {c['q']}\n    expected {c['expect']!r}, got {r['answer'][:90]!r}")
        dt = time.time() - t0
        results[mode] = hits / len(cases)
        print(f"{mode:22s} {hits:3d}/{len(cases)}  = {hits/len(cases)*100:5.1f}%   "
              f"({dt/len(cases):.2f}s per question)")

    print()
    for mode, acc in results.items():
        print(f"{mode:22s} {'#' * int(acc * 40)} {acc*100:.1f}%")


if __name__ == "__main__":
    main()
