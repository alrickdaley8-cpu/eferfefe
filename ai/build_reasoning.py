"""Reasoning data for the tiny LM: every answer is preceded by explicit worked steps.

A 5M-parameter model cannot do latent multi-step reasoning inside its activations — it has ~4
layers to work with. What it *can* learn is to externalise the steps into tokens and then read
its own steps, which is what makes small models measurably better at multi-hop questions.

Format (loss is taken on the whole answer, thinking included):

    ### Question:
    Context: fastapi 0.115.0: … Dependencies: pydantic, starlette, typing-extensions.
    Question: Does fastapi depend on pydantic?

    ### Answer:
    <think>
    Dependencies listed: pydantic, starlette, typing-extensions.
    Looking for pydantic. Found it in the list.
    </think>
    Yes, fastapi depends on pydantic.

Generates from ai/data/qa/knowledge.jsonl (no network needed):

    python ai/build_reasoning.py --out ai/data/qa/reasoning.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_DIR = os.path.join(ROOT, "ai", "data", "qa")

THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n"


def think(steps: list[str], answer: str) -> str:
    return THINK_OPEN + "\n".join(steps) + THINK_CLOSE + answer


def context_of(k: dict) -> str:
    bits = [f"{k['name']} {k.get('version','')}: {k.get('summary','')}".strip()]
    if k.get("description") and k["description"] != k.get("summary"):
        bits.append(k["description"])
    if k.get("license"):
        bits.append(f"License: {k['license']}.")
    if k.get("requires_python"):
        bits.append(f"Requires Python {k['requires_python']}.")
    if k.get("author"):
        bits.append(f"Author: {k['author']}.")
    if k.get("deps"):
        bits.append("Dependencies: " + ", ".join(k["deps"]) + ".")
    if k.get("topics"):
        bits.append("Topics: " + ", ".join(k["topics"]) + ".")
    if k.get("home"):
        bits.append(f"Homepage: {k['home']}.")
    return " ".join(bits)


# ======================================================================================
# arithmetic with worked steps
# ======================================================================================
def add_steps(a: int, b: int) -> list[str]:
    sa, sb = str(a)[::-1], str(b)[::-1]
    carry, steps, digits = 0, [], []
    names = ["units", "tens", "hundreds", "thousands"]
    for i in range(max(len(sa), len(sb))):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        tot = da + db + carry
        steps.append(f"{names[min(i, 3)]}: {da} + {db}"
                     + (f" + carry {carry}" if carry else "")
                     + f" = {tot}, write {tot % 10}" + (", carry 1" if tot >= 10 else ""))
        digits.append(str(tot % 10))
        carry = tot // 10
    if carry:
        steps.append(f"final carry {carry}")
        digits.append(str(carry))
    steps.append("reading the digits back: " + "".join(reversed(digits)))
    return steps


def arithmetic(rng: random.Random, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "add", "sub", "mul", "compare", "percent", "avg", "seq"])
        if kind == "add":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            out.append({"prompt": f"What is {a} + {b}?",
                        "response": think(add_steps(a, b), f"{a} + {b} = {a + b}")})
        elif kind == "sub":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            hi, lo = max(a, b), min(a, b)
            steps = [f"{a} - {b}: the larger number is {hi}.",
                     f"{hi} - {lo} = {hi - lo}."]
            if b > a:
                steps.append("the answer is negative because we subtracted a larger number.")
            out.append({"prompt": f"What is {a} - {b}?",
                        "response": think(steps, f"{a} - {b} = {a - b}")})
        elif kind == "mul":
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            tens, units = (b // 10) * 10, b % 10
            steps = [f"split {b} into {tens} + {units}.",
                     f"{a} x {tens} = {a * tens}.", f"{a} x {units} = {a * units}.",
                     f"add them: {a * tens} + {a * units} = {a * b}."]
            out.append({"prompt": f"What is {a} times {b}?",
                        "response": think(steps, f"{a} * {b} = {a * b}")})
        elif kind == "compare":
            a, b = rng.randint(1, 99999), rng.randint(1, 99999)
            steps = [f"{a} has {len(str(a))} digits, {b} has {len(str(b))} digits."]
            steps.append("more digits means larger." if len(str(a)) != len(str(b))
                         else "same digit count, so compare from the left.")
            big = max(a, b)
            out.append({"prompt": f"Which is bigger, {a} or {b}?",
                        "response": think(steps, f"{big} is bigger.")})
        elif kind == "percent":
            p, v = rng.choice([10, 20, 25, 50]), rng.randint(20, 800)
            steps = [f"{p}% means {p}/100.", f"{v} x {p} = {v * p}.", f"{v * p} / 100 = {v * p / 100:g}."]
            out.append({"prompt": f"What is {p}% of {v}?",
                        "response": think(steps, f"{p}% of {v} is {v * p / 100:g}.")})
        elif kind == "avg":
            xs = [rng.randint(1, 200) for _ in range(rng.randint(3, 4))]
            steps = [f"sum: {' + '.join(map(str, xs))} = {sum(xs)}.",
                     f"count: {len(xs)} numbers.",
                     f"{sum(xs)} / {len(xs)} = {sum(xs)/len(xs):g}."]
            out.append({"prompt": f"What is the average of {', '.join(map(str, xs))}?",
                        "response": think(steps, f"The average is {sum(xs)/len(xs):g}.")})
        else:
            start, stepv = rng.randint(1, 20), rng.randint(2, 9)
            seq = [start + i * stepv for i in range(5)]
            steps = [f"differences: {seq[1]-seq[0]}, {seq[2]-seq[1]}, {seq[3]-seq[2]}, {seq[4]-seq[3]}.",
                     f"the step is constant at {stepv}.", f"{seq[-1]} + {stepv} = {seq[-1]+stepv}."]
            out.append({"prompt": "What comes next in this sequence: "
                                  + ", ".join(map(str, seq)) + "?",
                        "response": think(steps, f"The next number is {seq[-1] + stepv}.")})
    return out


# ======================================================================================
# string / counting reasoning
# ======================================================================================
WORDS = ["package", "python", "installer", "formatter", "requests", "parser", "wheel", "module",
         "runtime", "template", "database", "serializer", "compiler", "notebook", "pipeline"]


def strings(rng: random.Random, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        w = rng.choice(WORDS)
        kind = rng.choice(["count_letter", "length", "reverse", "first_last", "vowels"])
        if kind == "count_letter":
            ch = rng.choice(sorted(set(w)))
            hits = [str(i + 1) for i, c in enumerate(w) if c == ch]
            steps = ["spelling it out: " + " ".join(w),
                     f"positions holding '{ch}': {', '.join(hits)}."]
            out.append({"prompt": f"How many times does the letter {ch} appear in {w}?",
                        "response": think(steps, f"The letter {ch} appears {len(hits)} time"
                                                 f"{'s' if len(hits) != 1 else ''} in {w}.")})
        elif kind == "length":
            steps = ["counting: " + " ".join(f"{c}{i+1}" for i, c in enumerate(w)) + "."]
            out.append({"prompt": f"How many letters are in the word {w}?",
                        "response": think(steps, f"{w} has {len(w)} letters.")})
        elif kind == "reverse":
            steps = ["reading it backwards: " + " ".join(reversed(w)) + "."]
            out.append({"prompt": f"Reverse the string {w}.", "response": think(steps, w[::-1])})
        elif kind == "vowels":
            vs = [c for c in w if c in "aeiou"]
            steps = ["letters: " + " ".join(w), f"vowels found: {', '.join(vs)}."]
            out.append({"prompt": f"How many vowels are in {w}?",
                        "response": think(steps, f"{w} has {len(vs)} vowels.")})
        else:
            steps = [f"first character: {w[0]}.", f"last character: {w[-1]}."]
            out.append({"prompt": f"What are the first and last letters of {w}?",
                        "response": think(steps, f"The first letter is {w[0]} and the last is {w[-1]}.")})
    return out


# ======================================================================================
# grounded single-hop: read the field out of the context, then answer
# ======================================================================================
def grounded(k: dict, rng: random.Random) -> list[dict]:
    ctx, n = context_of(k), k["name"]
    out = []

    def q(question: str, steps: list[str], answer: str) -> None:
        out.append({"prompt": f"Context: {ctx}\nQuestion: {question}",
                    "response": think(steps, answer)})

    if k.get("license"):
        q(f"What license does {n} use?",
          [f"the context is about {n}.", f"it says: License: {k['license']}."],
          f"{n} is released under the {k['license']} license.")
    if k.get("requires_python"):
        q(f"Which Python version does {n} need?",
          [f"the context is about {n}.", f"it says: Requires Python {k['requires_python']}."],
          f"{n} requires Python {k['requires_python']}.")
    if k.get("version"):
        q(f"What version of {n} is current?",
          [f"the context is about {n}.", f"the version shown is {k['version']}."],
          f"The latest version of {n} is {k['version']}.")
    if k.get("deps"):
        deps = k["deps"]
        dep = rng.choice(deps)
        q(f"Does {n} depend on {dep}?",
          ["dependencies listed: " + ", ".join(deps) + ".",
           f"looking for {dep}: it is in the list."],
          f"Yes, {n} depends on {dep}.")
        q(f"How many dependencies does {n} have?",
          ["dependencies listed: " + ", ".join(deps) + ".",
           "counting them: " + ", ".join(f"{d}={i+1}" for i, d in enumerate(deps)) + "."],
          f"{n} has {len(deps)} listed dependencies.")
        if rng.random() < 0.6:
            fake = rng.choice(["tensorflow", "django", "pygame", "scrapy", "matplotlib"])
            if fake not in deps:
                q(f"Does {n} depend on {fake}?",
                  ["dependencies listed: " + ", ".join(deps) + ".",
                   f"looking for {fake}: it is not in the list."],
                  f"No, {fake} is not among the listed dependencies of {n}.")
    q(f"What is {n} for?",
      [f"the summary says: {k.get('summary','')[:150]}"],
      f"{n} is a Python package. {k.get('summary','')}")
    return out


# ======================================================================================
# multi-hop: two contexts, one comparison
# ======================================================================================
def semver(v: str) -> tuple:
    parts = re.findall(r"\d+", v)[:3]
    return tuple(int(p) for p in parts) if parts else (0,)


def multihop(a: dict, b: dict, rng: random.Random) -> list[dict]:
    ctx = f"(1) {context_of(a)}\n(2) {context_of(b)}"
    out = []

    def q(question: str, steps: list[str], answer: str) -> None:
        out.append({"prompt": f"Context: {ctx}\nQuestion: {question}",
                    "response": think(steps, answer)})

    na, nb = a["name"], b["name"]
    if a.get("license") and b.get("license"):
        same = a["license"] == b["license"]
        q(f"Do {na} and {nb} use the same license?",
          [f"{na} license: {a['license']}.", f"{nb} license: {b['license']}.",
           "the two licenses are " + ("identical." if same else "different.")],
          (f"Yes, both use the {a['license']} license." if same else
           f"No. {na} uses {a['license']} and {nb} uses {b['license']}."))
    if a.get("deps") and b.get("deps"):
        more = na if len(a["deps"]) >= len(b["deps"]) else nb
        q(f"Which has more dependencies, {na} or {nb}?",
          [f"{na} lists {len(a['deps'])}: " + ", ".join(a["deps"]) + ".",
           f"{nb} lists {len(b['deps'])}: " + ", ".join(b["deps"]) + ".",
           f"{len(a['deps'])} vs {len(b['deps'])}."],
          f"{more} has more listed dependencies.")
        shared = sorted(set(a["deps"]) & set(b["deps"]))
        q(f"Do {na} and {nb} share any dependency?",
          [f"{na}: " + ", ".join(a["deps"]) + ".", f"{nb}: " + ", ".join(b["deps"]) + ".",
           ("common entries: " + ", ".join(shared) + ".") if shared else "no overlap between the lists."],
          (f"Yes, they both depend on {', '.join(shared)}." if shared
           else f"No, {na} and {nb} have no dependency in common."))
    if a.get("version") and b.get("version"):
        newer = na if semver(a["version"]) >= semver(b["version"]) else nb
        q(f"Which is on a higher version number, {na} or {nb}?",
          [f"{na} is at {a['version']}.", f"{nb} is at {b['version']}.",
           "comparing the numbers from the left."],
          f"{newer} has the higher version number.")
    if a.get("requires_python") and b.get("requires_python") and \
            a["requires_python"] != b["requires_python"]:
        q(f"Which needs the newer Python, {na} or {nb}?",
          [f"{na} requires Python {a['requires_python']}.",
           f"{nb} requires Python {b['requires_python']}.",
           "comparing the minimum versions."],
          f"{na if semver(a['requires_python']) >= semver(b['requires_python']) else nb} "
          f"requires the newer Python.")
    # choose-the-right-package (retrieval reranking, with steps)
    if a.get("summary") and b.get("summary") and rng.random() < 0.5:
        q(f"I need {a['summary'][:70].lower()}. Which of these two should I use?",
          [f"(1) {na}: {a['summary'][:90]}", f"(2) {nb}: {nb and b['summary'][:90]}",
           "the first one matches the requirement."],
          f"Use {na}. {a['summary']}")
    # unanswerable -> refuse, with the check shown
    if rng.random() < 0.3:
        q(f"How many downloads does {na} have?",
          ["scanning the context for download counts.", "no download figure is present."],
          "The context does not say, so I cannot answer that reliably.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", default=os.path.join(QA_DIR, "knowledge.jsonl"))
    ap.add_argument("--out", default=os.path.join(QA_DIR, "reasoning.jsonl"))
    ap.add_argument("--packages", type=int, default=8000, help="packages used for grounded items")
    ap.add_argument("--pairs", type=int, default=12000, help="two-package comparison items")
    ap.add_argument("--arith", type=int, default=30000)
    ap.add_argument("--strings", type=int, default=12000)
    args = ap.parse_args()

    know = [json.loads(l) for l in open(args.knowledge)]
    rng = random.Random(11)
    rng.shuffle(know)
    print(f"[reason] {len(know):,} knowledge records")

    examples: list[dict] = []
    for k in know[: args.packages]:
        examples.extend(grounded(k, rng))
    for _ in range(args.pairs):
        a, b = rng.sample(know, 2)
        examples.extend(multihop(a, b, rng))
    examples.extend(arithmetic(rng, args.arith))
    examples.extend(strings(rng, args.strings))
    rng.shuffle(examples)

    os.makedirs(QA_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    chars = sum(len(e["prompt"]) + len(e["response"]) for e in examples)
    print(f"[reason] {len(examples):,} reasoning examples, ~{chars/1e6:.1f}M chars "
          f"(~{chars/3.6/1e6:.1f}M tokens) -> {args.out}")


if __name__ == "__main__":
    main()
