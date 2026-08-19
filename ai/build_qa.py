"""Build an instruction / question-answering dataset for the tiny LM.

A 5M-parameter model cannot memorise world knowledge, so we train it to do the thing a model
this size *can* learn: follow a chat format and answer questions **from a context passage that
is retrieved for it at inference time** (plus a small amount of closed-book knowledge about the
PyPI ecosystem, and honest refusals when the context does not contain the answer).

Sources reachable from this sandbox: the PyPI JSON API (package metadata) and the corpus itself.

Output:
    ai/data/qa/knowledge.jsonl   one record per package -> also used as the retrieval index
    ai/data/qa/sft.jsonl         {"prompt": ..., "response": ...} chat examples

Usage:
    python ai/build_qa.py --packages 12000 --workers 32
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
QA_DIR = os.path.join(DATA, "qa")
TOP_CSV = os.path.join(DATA, "top-pypi-packages.csv")

USER, ASSIST, END = "<|user|>", "<|assistant|>", "<|endoftext|>"
_lock = threading.Lock()


def log(m: str) -> None:
    sys.stderr.write(m + "\n")
    sys.stderr.flush()


def clean_text(s: str | None, limit: int = 400) -> str:
    if not s:
        return ""
    s = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)|!\[.*?\]\(.*?\)", " ", s)   # badges/images
    s = re.sub(r"<[^>\n]{0,200}>", " ", s)                              # html tags
    s = re.sub(r"<[a-z][^\s>]*", " ", s)                                # unclosed html tags
    s = re.sub(r"https?://\S*(badge|shields|svg|actions|readthedocs)\S*", " ", s)  # badge urls
    s = re.sub(r"[-=]{3,}", " ", s)
    s = re.sub(r"[`*_#>|]+", " ", s)
    s = re.sub(r"https?://\S+", lambda m: m.group(0)[:60], s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit].rsplit(" ", 1)[0] if len(s) > limit else s


def first_sentences(s: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", s)
    return " ".join(parts[:n]).strip()


# --------------------------------------------------------------------------------------
# knowledge extraction
# --------------------------------------------------------------------------------------
def fetch_pkg(name: str, sess: requests.Session) -> dict | None:
    try:
        r = sess.get(f"https://pypi.org/pypi/{name}/json", timeout=20)
        if r.status_code != 200:
            return None
        info = r.json()["info"]
    except Exception:
        return None

    summary = clean_text(info.get("summary"), 200)
    desc = clean_text(first_sentences(clean_text(info.get("description"), 1200), 2), 220)
    if not summary and not desc:
        return None
    deps = [re.split(r"[<>=!;\[ ]", d)[0] for d in (info.get("requires_dist") or [])]
    deps = sorted({d for d in deps if d})[:8]
    classifiers = info.get("classifiers") or []
    topics = [c.split("::")[-1].strip() for c in classifiers if c.startswith("Topic ::")][:4]
    lic = clean_text(info.get("license_expression") or info.get("license") or "", 40)
    if not lic:
        lic = next((c.split("::")[-1].strip() for c in classifiers if c.startswith("License ::")), "")
    home = info.get("home_page") or (info.get("project_urls") or {}).get("Homepage") or ""
    return {
        "name": info.get("name") or name,
        "summary": summary or desc,
        "description": desc,
        "version": info.get("version") or "",
        "author": clean_text(info.get("author") or info.get("author_email") or "", 60),
        "license": lic,
        "requires_python": info.get("requires_python") or "",
        "deps": deps,
        "topics": topics,
        "home": home[:80],
        "keywords": clean_text(info.get("keywords") or "", 80),
    }


# --------------------------------------------------------------------------------------
# example generation
# --------------------------------------------------------------------------------------
def context_of(k: dict) -> str:
    bits = [f"{k['name']} {k['version']}: {k['summary']}"]
    if k["description"] and k["description"] != k["summary"]:
        bits.append(k["description"])
    if k["license"]:
        bits.append(f"License: {k['license']}.")
    if k["requires_python"]:
        bits.append(f"Requires Python {k['requires_python']}.")
    if k["author"]:
        bits.append(f"Author: {k['author']}.")
    if k["deps"]:
        bits.append("Dependencies: " + ", ".join(k["deps"]) + ".")
    if k["topics"]:
        bits.append("Topics: " + ", ".join(k["topics"]) + ".")
    if k["home"]:
        bits.append(f"Homepage: {k['home']}.")
    return " ".join(bits)


ASK_WHAT = ["What is {n}?", "What does {n} do?", "Tell me about {n}.", "Describe the {n} package.",
            "What is the Python package {n} used for?", "Explain what {n} is."]
ASK_INSTALL = ["How do I install {n}?", "How can I install {n} with pip?",
               "What command installs {n}?", "Give me the install command for {n}."]
ASK_VERSION = ["What is the latest version of {n}?", "Which version of {n} is current?",
               "What version is {n} on?"]
ASK_LICENSE = ["What license does {n} use?", "Under which license is {n} released?",
               "Is {n} open source, and under what license?"]
ASK_PY = ["Which Python versions does {n} support?", "What Python version do I need for {n}?"]
ASK_DEPS = ["What are the dependencies of {n}?", "What does {n} depend on?",
            "Which packages does {n} require?"]
ASK_AUTHOR = ["Who wrote {n}?", "Who is the author of {n}?", "Who maintains {n}?"]
ASK_HOME = ["Where can I find {n}'s homepage?", "What is the project URL for {n}?"]
ASK_IMPORT = ["How do I import {n} in Python?", "What is the import statement for {n}?"]

GENERIC_PY = [
    ("How do I read a text file in Python?",
     "Use a context manager:\n\nwith open(\"file.txt\", \"r\", encoding=\"utf-8\") as f:\n    text = f.read()\n\nThe file is closed automatically when the block ends."),
    ("How do I write JSON to a file in Python?",
     "Use the json module:\n\nimport json\n\nwith open(\"data.json\", \"w\") as f:\n    json.dump(data, f, indent=2)"),
    ("How do I loop over a dictionary in Python?",
     "Iterate over items():\n\nfor key, value in d.items():\n    print(key, value)"),
    ("How do I create a virtual environment?",
     "Run python -m venv .venv, then activate it with source .venv/bin/activate on Linux or macOS, or .venv\\Scripts\\activate on Windows."),
    ("How do I install a package with pip?",
     "Run pip install <package>. Add --upgrade to update it, or pip install -r requirements.txt to install from a requirements file."),
    ("What is a list comprehension?",
     "A compact way to build a list from an iterable, for example squares = [x * x for x in range(10)]. You can add a condition: evens = [x for x in nums if x % 2 == 0]."),
    ("How do I handle exceptions in Python?",
     "Wrap the risky code in try / except:\n\ntry:\n    value = int(text)\nexcept ValueError as exc:\n    print(\"bad number:\", exc)"),
    ("What does __init__.py do?",
     "It marks a directory as a Python package and runs when the package is first imported, which is where packages usually expose their public API."),
    ("How do I sort a list of dictionaries by a key?",
     "Use the key argument: items.sort(key=lambda d: d[\"name\"]), or sorted(items, key=lambda d: d[\"name\"], reverse=True)."),
    ("What is the difference between a list and a tuple?",
     "A list is mutable and written with square brackets; a tuple is immutable and written with parentheses, so it can be used as a dictionary key."),
    ("How do I make an HTTP request in Python?",
     "With requests:\n\nimport requests\n\nresponse = requests.get(\"https://example.com\", timeout=10)\nprint(response.status_code, response.text)"),
    ("How do I run tests with pytest?",
     "Run pytest from the project root. It collects files named test_*.py and functions named test_*. Use pytest -k name to run a subset and pytest -q for quiet output."),
]

REFUSALS = [
    "Who won the 2038 World Cup?", "What is the capital of Mars?",
    "What will the stock market do tomorrow?", "What am I thinking right now?",
    "What is my phone number?", "Who is the president in the year 3000?",
    "How many people live in my street?", "What did I eat for breakfast?",
]
REFUSE_A = ("I do not know. I am a 5M parameter model trained on Python packaging text, so I can "
            "only answer from the context I am given or from what I learned about Python packages.")


def make_examples(k: dict, rng: random.Random, others: list[dict]) -> list[dict]:
    n = k["name"]
    ctx = context_of(k)
    ex: list[dict] = []

    def add(q: str, a: str, with_ctx: bool = True) -> None:
        if not a or len(a) < 3:
            return
        prompt = f"Context: {ctx}\nQuestion: {q}" if with_ctx else q
        ex.append({"prompt": prompt, "response": a})

    # grounded (retrieval-style) QA -- the main skill
    add(rng.choice(ASK_WHAT).format(n=n), f"{n} is a Python package. {k['summary']}")
    add(rng.choice(ASK_INSTALL).format(n=n), f"Install it with pip:\n\npip install {n.lower()}")
    if k["version"]:
        add(rng.choice(ASK_VERSION).format(n=n), f"The latest version of {n} is {k['version']}.")
    if k["license"]:
        add(rng.choice(ASK_LICENSE).format(n=n), f"{n} is released under the {k['license']} license.")
    if k["requires_python"]:
        add(rng.choice(ASK_PY).format(n=n), f"{n} requires Python {k['requires_python']}.")
    if k["deps"]:
        add(rng.choice(ASK_DEPS).format(n=n), f"{n} depends on {', '.join(k['deps'])}.")
    if k["author"]:
        add(rng.choice(ASK_AUTHOR).format(n=n), f"{n} is maintained by {k['author']}.")
    if k["home"]:
        add(rng.choice(ASK_HOME).format(n=n), f"The project page for {n} is {k['home']}.")
    add(rng.choice(ASK_IMPORT).format(n=n),
        f"import {re.sub(r'[^a-z0-9_]', '_', n.lower())}")
    if k["topics"]:
        add(f"What kind of library is {n}?",
            f"{n} is a {', '.join(k['topics']).lower()} library. {k['summary']}")

    # closed-book variants (no context) for the most memorisable facts
    if rng.random() < 0.5:
        ex.append({"prompt": rng.choice(ASK_WHAT).format(n=n),
                   "response": f"{n} is a Python package. {k['summary']}"})
    if rng.random() < 0.5:
        ex.append({"prompt": rng.choice(ASK_INSTALL).format(n=n),
                   "response": f"Install it with pip:\n\npip install {n.lower()}"})

    # recommendation direction: description -> package name
    if k["summary"] and rng.random() < 0.4:
        ex.append({"prompt": f"Context: {ctx}\nQuestion: Which package should I use if I need "
                             f"{k['summary'][:90].lower()}?",
                   "response": f"Use {n}. {k['summary']}"})

    # unanswerable-from-context -> honest refusal (grounding discipline)
    if rng.random() < 0.35:
        other = rng.choice(others)
        ex.append({"prompt": f"Context: {ctx}\nQuestion: What is {other['name']}?",
                   "response": "The context does not mention that, so I cannot answer it reliably."})
    return ex


def arithmetic_examples(rng: random.Random, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "count", "upper", "rev"])
        if kind == "add":
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            out.append({"prompt": f"What is {a} + {b}?", "response": f"{a} + {b} = {a + b}"})
        elif kind == "sub":
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            out.append({"prompt": f"What is {a} - {b}?", "response": f"{a} - {b} = {a - b}"})
        elif kind == "mul":
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            out.append({"prompt": f"What is {a} times {b}?", "response": f"{a} * {b} = {a * b}"})
        else:
            w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8)))
            if kind == "count":
                out.append({"prompt": f"How many letters are in the word {w}?",
                            "response": f"The word {w} has {len(w)} letters."})
            elif kind == "upper":
                out.append({"prompt": f"Write {w} in uppercase.", "response": w.upper()})
            else:
                out.append({"prompt": f"Reverse the string {w}.", "response": w[::-1]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--arith", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(QA_DIR, exist_ok=True)

    with open(TOP_CSV) as f:
        names = [r["project"] for r in csv.DictReader(f)][: args.packages]

    sess = requests.Session()
    sess.headers["User-Agent"] = "tiny-lm-qa-builder/1.0"
    know: list[dict] = []
    t0 = time.time()

    def work(nm: str) -> None:
        k = fetch_pkg(nm, sess)
        if k:
            with _lock:
                know.append(k)
                if len(know) % 500 == 0:
                    log(f"[qa] metadata {len(know)}/{len(names)}  {time.time()-t0:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, names))

    with open(os.path.join(QA_DIR, "knowledge.jsonl"), "w") as f:
        for k in know:
            f.write(json.dumps(k) + "\n")
    log(f"[qa] {len(know)} packages of knowledge")

    rng = random.Random(0)
    examples: list[dict] = []
    for k in know:
        examples.extend(make_examples(k, rng, know))
    for q, a in GENERIC_PY:
        examples.extend([{"prompt": q, "response": a}] * 40)          # upweight core how-tos
    for q in REFUSALS:
        examples.extend([{"prompt": q, "response": REFUSE_A}] * 40)
    examples.extend(arithmetic_examples(rng, args.arith))
    rng.shuffle(examples)

    with open(os.path.join(QA_DIR, "sft.jsonl"), "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    chars = sum(len(e["prompt"]) + len(e["response"]) for e in examples)
    log(f"[qa] {len(examples):,} SFT examples, ~{chars/1e6:.1f}M chars "
        f"(~{chars/3.6/1e6:.1f}M tokens) -> {QA_DIR}/sft.jsonl")


if __name__ == "__main__":
    main()
