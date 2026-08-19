"""Keyword retrieval over the PyPI knowledge base (ai/data/qa/knowledge.jsonl).

A 5M-parameter model has almost no room to memorise facts, so at inference time we look up the
relevant package record and put it in the prompt as `Context:` — the same shape the model was
fine-tuned on. Pure stdlib BM25-lite: no extra dependencies.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "ai", "data", "qa", "knowledge.jsonl")

_TOKEN = re.compile(r"[a-z0-9_]+")


def norm(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class Retriever:
    def __init__(self, path: str = KB_PATH):
        self.docs: list[dict] = []
        self.by_name: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    self.docs.append(d)
                    self.by_name[d["name"].lower()] = d
        self.tf: list[Counter] = []
        self.len: list[int] = []
        df: Counter = Counter()
        for d in self.docs:
            toks = norm(self.text(d))
            c = Counter(toks)
            self.tf.append(c)
            self.len.append(max(1, len(toks)))
            df.update(c.keys())
        self.N = max(1, len(self.docs))
        self.avg = sum(self.len) / self.N if self.docs else 1.0
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    @staticmethod
    def text(d: dict) -> str:
        return " ".join([d["name"], d.get("summary", ""), d.get("description", ""),
                         d.get("keywords", ""), " ".join(d.get("topics", []))])

    @staticmethod
    def context_of(d: dict) -> str:
        bits = [f"{d['name']} {d.get('version','')}: {d.get('summary','')}".strip()]
        if d.get("description") and d["description"] != d.get("summary"):
            bits.append(d["description"])
        if d.get("license"):
            bits.append(f"License: {d['license']}.")
        if d.get("requires_python"):
            bits.append(f"Requires Python {d['requires_python']}.")
        if d.get("author"):
            bits.append(f"Author: {d['author']}.")
        if d.get("deps"):
            bits.append("Dependencies: " + ", ".join(d["deps"]) + ".")
        if d.get("topics"):
            bits.append("Topics: " + ", ".join(d["topics"]) + ".")
        if d.get("home"):
            bits.append(f"Homepage: {d['home']}.")
        return " ".join(bits)

    def search(self, query: str, k: int = 1) -> list[tuple[float, dict]]:
        if not self.docs:
            return []
        q = norm(query)
        # exact package-name hit wins outright
        # ...but only for distinctive names: "parse" or "json" are ordinary English words,
        # so require the token to be rare in the corpus before treating it as a package name.
        name_bonus: dict[str, float] = {}
        for tok in q:
            if tok in self.by_name and len(tok) >= 3:
                d = self.by_name[tok]
                name_bonus[d["name"]] = max(name_bonus.get(d["name"], 0.0),
                                            6.0 * self.idf.get(tok, 1.0))
        scores = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = c.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * self.len[i] / self.avg))
            s += name_bonus.get(self.docs[i]["name"], 0.0)
            if s > 0:
                scores.append((s, self.docs[i]))
        scores.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for s, d in scores:
            if d["name"] in seen:
                continue
            seen.add(d["name"])
            out.append((s, d))
            if len(out) >= k:
                break
        return out

    def context_for(self, query: str, min_score: float = 1.5) -> str | None:
        res = self.search(query, k=1)
        if not res or res[0][0] < min_score:
            return None
        return self.context_of(res[0][1])


if __name__ == "__main__":
    import sys
    r = Retriever()
    print(f"{len(r.docs)} documents in knowledge base")
    q = " ".join(sys.argv[1:]) or "what library parses html"
    for s, d in r.search(q, k=3):
        print(f"{s:10.2f}  {d['name']}: {d.get('summary','')[:90]}")
