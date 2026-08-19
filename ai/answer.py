"""Deterministic answer layer — the part of the system that is allowed to be *certain*.

A 5M-parameter model is a good router and a decent phraser, but it is a terrible database and a
worse calculator. So the facts come from two places it cannot get wrong:

  * the retrieved package record (licence, version, dependencies, Python requirement, …)
  * a small calculator for arithmetic / string questions

`solve()` returns the grounded answer together with the same `<think>` steps the model was trained
to produce, so the shape of the reply is identical either way. `verify()` compares what the model
generated against the ground truth and reports whether it agrees — that is what lets the chat UI
label an answer "verified", "corrected" or "unchecked" instead of quietly bluffing.
"""
from __future__ import annotations

import re
from typing import Any

Doc = dict[str, Any]

# --------------------------------------------------------------------------------------
# conversational intents and Python how-tos
#
# These are the questions a person actually types first ("hi", "what can you do", "how do I read
# a file"). They have nothing to do with the knowledge base, and a 5M model answers them with
# noise, so they get real answers here.
# --------------------------------------------------------------------------------------
IDENTITY = ("I am tiny-lm: a 5,015,808 parameter transformer trained from scratch on 100M tokens "
            "of PyPI source text, running on CPU. I answer from a knowledge base of 11,922 Python "
            "packages, plus a calculator for arithmetic.")
CAPABILITIES = (
    "I can answer questions about Python packages — what a package does, its licence, version, "
    "dependencies, author, homepage, install and import commands, and comparisons between two "
    "packages. I can also do arithmetic, counting and string questions, and answer common Python "
    "how-tos. Ask me something like \"what license does black use?\" or \"does fastapi depend on "
    "pydantic?\" or \"what is 3471 + 2856?\".")

INTENTS: list[tuple[str, str, list[str]]] = [
    (r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening))\b",
     "Hello. Ask me about a Python package, or give me some arithmetic to do.",
     ["a greeting, so no lookup is needed."]),
    (r"\b(who|what) are you\b|\byour name\b|\bwhat model are you\b|\babout yourself\b",
     IDENTITY, ["the question is about me rather than a package."]),
    (r"\bwhat can you do\b|\bhelp\b|\bwhat do you know\b|\bhow do you work\b|"
     r"\bwhat (kind of )?questions\b",
     CAPABILITIES, ["listing what is actually inside the knowledge base."]),
    (r"\bhow (big|large|many parameters)\b|\bhow were you trained\b|\btraining data\b",
     "I have 5,015,808 parameters (4 layers, d_model 256, 8 heads, 8,192-token vocabulary, "
     "512-token context) and was pretrained on 100,000,000 tokens of text collected from PyPI "
     "source releases, then instruction tuned on 288k question/answer and reasoning examples.",
     ["reporting my own configuration."]),
    (r"\b(thanks|thank you|cheers|ta)\b", "You are welcome.", ["a thank-you needs no lookup."]),
    (r"\b(bye|goodbye|see you)\b", "Goodbye.", ["a farewell needs no lookup."]),
]

HOWTOS: list[tuple[str, str]] = [
    (r"read (a )?(text )?file",
     "Use a context manager:\n\nwith open(\"file.txt\", encoding=\"utf-8\") as f:\n"
     "    text = f.read()\n\nThe file is closed automatically when the block ends."),
    (r"write (to )?(a )?file|save .* to (a )?file",
     "Open it in write mode:\n\nwith open(\"out.txt\", \"w\", encoding=\"utf-8\") as f:\n"
     "    f.write(text)"),
    (r"(read|parse|load).*json|json.*(file|parse)",
     "Use the json module:\n\nimport json\n\nwith open(\"data.json\") as f:\n"
     "    data = json.load(f)\n\nUse json.dump(data, f, indent=2) to write it back."),
    (r"virtual ?env|venv",
     "Create one with python -m venv .venv, then activate it: source .venv/bin/activate on Linux "
     "or macOS, .venv\\Scripts\\activate on Windows."),
    (r"install .*(package|library|module|dependenc)|use pip",
     "Run pip install <package>. Add --upgrade to update it, or pip install -r requirements.txt "
     "to install everything a project needs."),
    (r"list comprehension",
     "A compact way to build a list: squares = [x * x for x in range(10)]. Add a condition with "
     "if: evens = [x for x in nums if x % 2 == 0]."),
    (r"(handle|catch).*(exception|error)|try.*except",
     "Wrap the risky code:\n\ntry:\n    value = int(text)\nexcept ValueError as exc:\n"
     "    print(\"bad number:\", exc)"),
    (r"http request|call an api|fetch a url|download a (page|url)",
     "With requests:\n\nimport requests\n\nr = requests.get(\"https://example.com\", timeout=10)"
     "\nprint(r.status_code, r.text)"),
    (r"sort .*(list|dict)",
     "Use the key argument: items.sort(key=lambda d: d[\"name\"]), or "
     "sorted(items, key=..., reverse=True) to get a new list."),
    (r"difference between .*(list|tuple)",
     "A list is mutable and written with square brackets; a tuple is immutable, written with "
     "parentheses, and can be used as a dictionary key."),
    (r"run tests|use pytest",
     "Run pytest from the project root. It collects files named test_*.py and functions named "
     "test_*. pytest -k name runs a subset, pytest -q keeps the output short."),
    (r"__init__\.py",
     "It marks a directory as a package and runs on first import, which is where a package "
     "usually exposes its public API."),
    (r"loop over .*(dict|dictionary)",
     "Iterate over items():\n\nfor key, value in d.items():\n    print(key, value)"),
]


def conversational(q: str) -> dict | None:
    ql = q.lower().strip()
    for pattern, reply, steps in INTENTS:
        if re.search(pattern, ql):
            return {"answer": reply, "steps": steps, "kind": "assistant"}
    if re.search(r"\bhow (do|can) i\b|\bhow to\b|\bwhat is a\b|\bwhat does\b", ql) \
            or "python" in ql:
        for pattern, reply in HOWTOS:
            if re.search(pattern, ql):
                return {"answer": reply, "steps": ["this is a general Python question, "
                                                   "answered from the built-in how-tos."],
                        "kind": "python-howto"}
    return None


# --------------------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------------------
NUM = r"(-?\d[\d,]*)"


def _i(s: str) -> int:
    return int(s.replace(",", ""))


def calculator(q: str) -> dict | None:
    ql = q.lower().strip()

    m = re.search(rf"{NUM}\s*(?:\+|plus)\s*{NUM}", ql)
    if m:
        a, b = _i(m.group(1)), _i(m.group(2))
        return _calc(f"{a} + {b} = {a + b}", [f"{a} + {b}: adding column by column.",
                                              f"the total is {a + b}."])
    m = re.search(rf"{NUM}\s*(?:-|minus)\s*{NUM}", ql)
    if m:
        a, b = _i(m.group(1)), _i(m.group(2))
        return _calc(f"{a} - {b} = {a - b}", [f"{a} - {b}.", f"the difference is {a - b}."])
    m = re.search(rf"{NUM}\s*(?:\*|x|times|multiplied by)\s*{NUM}", ql)
    if m:
        a, b = _i(m.group(1)), _i(m.group(2))
        tens, units = (b // 10) * 10, b % 10
        return _calc(f"{a} * {b} = {a * b}",
                     [f"split {b} into {tens} + {units}.", f"{a} x {tens} = {a * tens}.",
                      f"{a} x {units} = {a * units}.", f"sum: {a * b}."])
    m = re.search(rf"{NUM}\s*(?:/|divided by)\s*{NUM}", ql)
    if m:
        a, b = _i(m.group(1)), _i(m.group(2))
        if b:
            return _calc(f"{a} / {b} = {a / b:g}", [f"dividing {a} by {b}."])
    m = re.search(rf"{NUM}\s*%\s*of\s*{NUM}", ql)
    if m:
        p, v = _i(m.group(1)), _i(m.group(2))
        return _calc(f"{p}% of {v} is {v * p / 100:g}",
                     [f"{p}% means {p}/100.", f"{v} x {p} = {v * p}.", f"divide by 100."])
    m = re.search(r"average of ([\d,\s]+(?:and)?[\d,\s]*)", ql)
    if m:
        xs = [_i(x) for x in re.findall(r"\d[\d,]*", m.group(1))]
        if len(xs) >= 2:
            return _calc(f"The average is {sum(xs)/len(xs):g}.",
                         [f"sum: {' + '.join(map(str, xs))} = {sum(xs)}.",
                          f"count: {len(xs)}.", f"{sum(xs)} / {len(xs)} = {sum(xs)/len(xs):g}."])
    m = re.search(r"which is bigger,?\s*" + NUM + r"\s*or\s*" + NUM, ql)
    if m:
        a, b = _i(m.group(1)), _i(m.group(2))
        return _calc(f"{max(a, b)} is bigger.",
                     [f"comparing {a} and {b}.", f"{max(a,b)} > {min(a,b)}."])
    m = re.search(r"(?:comes next|next number)[^:]*:?\s*([\d,\s]+)", ql)
    if m:
        xs = [_i(x) for x in re.findall(r"\d[\d,]*", m.group(1))]
        if len(xs) >= 3:
            diffs = {xs[i + 1] - xs[i] for i in range(len(xs) - 1)}
            if len(diffs) == 1:
                step = diffs.pop()
                return _calc(f"The next number is {xs[-1] + step}.",
                             [f"differences are all {step}.", f"{xs[-1]} + {step} = {xs[-1]+step}."])

    m = re.search(r"how many (?:letters|characters) (?:are )?in (?:the word )?['\"]?([a-z]+)", ql)
    if m:
        w = m.group(1)
        return _calc(f"{w} has {len(w)} letters.", ["counting: " + " ".join(w) + "."])
    m = re.search(r"how many (?:times )?(?:does the letter )?([a-z])\b.*\bin (?:the word )?"
                  r"['\"]?([a-z]{2,})", ql)
    if m:
        ch, w = m.group(1), m.group(2)
        n = w.count(ch)
        return _calc(f"The letter {ch} appears {n} time{'s' if n != 1 else ''} in {w}.",
                     ["spelling it out: " + " ".join(w) + ".", f"matches: {n}."])
    m = re.search(r"how many vowels (?:are )?in ['\"]?([a-z]+)", ql)
    if m:
        w = m.group(1)
        vs = [c for c in w if c in "aeiou"]
        return _calc(f"{w} has {len(vs)} vowels.", ["letters: " + " ".join(w) + ".",
                                                    "vowels: " + ", ".join(vs) + "."])
    m = re.search(r"reverse the (?:string|word) ['\"]?([a-z0-9_]+)", ql)
    if m:
        w = m.group(1)
        return _calc(w[::-1], ["reading it backwards."])
    return None


def _calc(answer: str, steps: list[str]) -> dict:
    return {"answer": answer, "steps": steps, "kind": "calculator"}


# --------------------------------------------------------------------------------------
# package facts
# --------------------------------------------------------------------------------------
def _fmt_deps(deps: list[str]) -> str:
    return ", ".join(deps)


def _subject(q: str, docs: list[Doc]) -> Doc:
    """Pick the package the question is *about* — the first one named in the text."""
    ql = q.lower()
    best, best_pos = docs[0], len(ql) + 1
    for d in docs:
        pos = ql.find(d["name"].lower())
        if pos >= 0 and pos < best_pos:
            best, best_pos = d, pos
    return best


PACKAGEY = re.compile(r"\b(package|library|module|pypi|pip|dependenc|import|install|version|"
                      r"licen[cs]e|maintainer|author|homepage)\b", re.I)
RECOMMEND = re.compile(r"which package|what should i use|recommend|what can i use", re.I)


def _about_a_package(q: str, docs: list[Doc]) -> bool:
    """Guard against answering "what is the capital of France?" with a random PyPI record."""
    ql = q.lower()
    if any(d["name"].lower() in ql for d in docs):
        return True
    return bool(PACKAGEY.search(ql) or RECOMMEND.search(ql))


def package_answer(q: str, docs: list[Doc]) -> dict | None:
    if not docs or not _about_a_package(q, docs):
        return None
    ql = q.lower()
    m_dep = re.search(r"does ([a-z0-9_.\-]+) depend on ([a-z0-9_.\-]+)", ql)
    if m_dep:
        named = [d for d in docs if d["name"].lower() == m_dep.group(1)]
        d = named[0] if named else _subject(q, docs)
    else:
        d = _subject(q, docs)
    n = d["name"]

    def ans(answer: str, steps: list[str]) -> dict:
        return {"answer": answer, "steps": steps, "kind": "knowledge-base", "package": n}

    # ---- two-package comparisons
    if len(docs) >= 2:
        a, b = docs[0], docs[1]
        na, nb = a["name"], b["name"]
        if "same license" in ql or ("license" in ql and (" and " in ql or " vs " in ql)):
            if a.get("license") and b.get("license"):
                same = a["license"].lower() == b["license"].lower()
                return ans(f"Yes, both use the {a['license']} license." if same else
                           f"No. {na} uses {a['license']} and {nb} uses {b['license']}.",
                           [f"{na} license: {a['license']}.", f"{nb} license: {b['license']}.",
                            "the licences are " + ("identical." if same else "different.")])
        if "more dependencies" in ql or ("dependencies" in ql and (" or " in ql or " vs " in ql)):
            la, lb = len(a.get("deps", [])), len(b.get("deps", []))
            return ans(f"{na if la >= lb else nb} has more listed dependencies "
                       f"({max(la, lb)} vs {min(la, lb)}).",
                       [f"{na} lists {la}.", f"{nb} lists {lb}."])
        if "share" in ql and "depend" in ql:
            shared = sorted(set(a.get("deps", [])) & set(b.get("deps", [])))
            return ans(f"Yes, they both depend on {_fmt_deps(shared)}." if shared else
                       f"No, {na} and {nb} have no listed dependency in common.",
                       [f"{na}: {_fmt_deps(a.get('deps', []))}.",
                        f"{nb}: {_fmt_deps(b.get('deps', []))}.",
                        ("common: " + _fmt_deps(shared)) if shared else "no overlap."])

    # ---- "does X depend on Y?"
    m = m_dep or re.search(r"depends? on ([a-z0-9_.\-]+)", ql)
    if m and d.get("deps") is not None:
        target = m.group(m.lastindex).strip(" ?.")
        deps = d.get("deps", [])
        hit = next((x for x in deps if x.lower() == target), None)
        return ans(f"Yes, {n} depends on {hit}." if hit else
                   f"No, {target} is not among the listed dependencies of {n}"
                   + (f" ({_fmt_deps(deps)})." if deps else "."),
                   [f"dependencies listed: {_fmt_deps(deps) or 'none'}.",
                    f"looking for {target}: " + ("found." if hit else "not in the list.")])

    if "how many dep" in ql and d.get("deps") is not None:
        deps = d["deps"]
        return ans(f"{n} lists {len(deps)} dependencies: {_fmt_deps(deps)}." if deps else
                   f"{n} lists no dependencies.",
                   [f"dependencies: {_fmt_deps(deps) or 'none'}.", f"count: {len(deps)}."])

    if re.search(r"python (version|release)|version of python|python.*(need|require|support)", ql) \
            and d.get("requires_python"):
        return ans(f"{n} requires Python {d['requires_python']}.",
                   [f"the context is about {n}.",
                    f"it says: Requires Python {d['requires_python']}."])
    if "licen" in ql and d.get("license"):
        return ans(f"{n} is released under the {d['license']} license.",
                   [f"the context is about {n}.", f"it says: License: {d['license']}."])
    if ("version" in ql or "latest" in ql) and d.get("version"):
        return ans(f"The latest version of {n} is {d['version']}.",
                   [f"the context is about {n}.", f"the version shown is {d['version']}."])
    if "python" in ql and ("version" in ql or "need" in ql or "support" in ql) \
            and d.get("requires_python"):
        return ans(f"{n} requires Python {d['requires_python']}.",
                   [f"the context is about {n}.",
                    f"it says: Requires Python {d['requires_python']}."])
    if any(w in ql for w in ("who wrote", "author", "maintains", "maintainer")) and d.get("author"):
        return ans(f"{n} is maintained by {d['author']}.",
                   [f"the context is about {n}.", f"author field: {d['author']}."])
    if ("homepage" in ql or "project url" in ql or "where can i find" in ql) and d.get("home"):
        return ans(f"The project page for {n} is {d['home']}.",
                   [f"the context is about {n}.", f"homepage field: {d['home']}."])
    if "depend" in ql and d.get("deps"):
        return ans(f"{n} depends on {_fmt_deps(d['deps'])}.",
                   [f"dependencies listed: {_fmt_deps(d['deps'])}."])
    if "install" in ql:
        return ans(f"Install it with pip:\n\npip install {n.lower()}",
                   [f"the package name on PyPI is {n.lower()}."])
    if "import" in ql:
        mod = re.sub(r"[^a-z0-9_]", "_", n.lower())
        return ans(f"import {mod}", [f"the distribution is {n}; the import name is usually {mod}."])
    if any(p in ql for p in ("what is", "what does", "tell me about", "describe", "what kind",
                             "used for", "what for")) and d.get("summary"):
        extra = f" {d['description']}" if d.get("description") and \
            d["description"] != d["summary"] else ""
        return ans(f"{n} is a Python package. {d['summary']}{extra}".strip(),
                   [f"the summary field says: {d['summary']}"])
    if any(p in ql for p in ("which package", "what should i use", "recommend")) and d.get("summary"):
        return ans(f"Use {n}. {d['summary']}",
                   [f"best match in the knowledge base: {n}.", f"summary: {d['summary']}"])
    return None


def solve(question: str, docs: list[Doc]) -> dict | None:
    """Ground-truth answer for a question, or None when nothing can be established."""
    return conversational(question) or calculator(question) or package_answer(question, docs)


# --------------------------------------------------------------------------------------
# quality gate for ungrounded output
# --------------------------------------------------------------------------------------
FALLBACK = ("I do not know that one. I am a 5M parameter model that answers from a knowledge base "
            "of 11,922 Python packages plus a calculator — try asking about a package "
            "(\"what license does black use?\"), a comparison (\"does fastapi depend on "
            "pydantic?\"), or some arithmetic.")


def looks_degenerate(text: str) -> bool:
    """True when generation collapsed into repetition or symbol soup rather than an answer."""
    t = text.strip()
    if len(t) < 3:
        return True
    words = re.findall(r"[a-zA-Z]{2,}", t)
    if len(words) < 3:
        return True
    if len(set(w.lower() for w in words)) / len(words) < 0.45:      # loops on a few words
        return True
    letters = sum(c.isalpha() or c.isspace() for c in t)
    if letters / len(t) < 0.55:                                     # mostly punctuation/digits
        return True
    if re.search(r"(.{2,12}?)\1{3,}", t):                            # "0.0.0.0.0.0"
        return True
    return False


# --------------------------------------------------------------------------------------
# verification of what the model said
# --------------------------------------------------------------------------------------
def _key_facts(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{2,}|\d+(?:\.\d+)*", text.lower()))


REFUSAL_MARKERS = ("do not know", "don't know", "cannot answer", "can not answer",
                   "does not mention", "not sure")


def _supported_by(answer: str, context: str) -> bool:
    """Do the answer's content words actually appear in the retrieved context?"""
    ctx = _key_facts(context)
    words = [w for w in _key_facts(answer) if len(w) > 3 or any(c.isdigit() for c in w)]
    if not words:
        return False
    return len([w for w in words if w in ctx]) / len(words) >= 0.5


def verify(model_answer: str, truth: dict | None, has_context: bool = True,
           context: str | None = None) -> dict:
    """Decide what the user actually sees.

    Policy: when a deterministic answer exists it *is* the answer — the model only gets to phrase
    things it demonstrably agrees with. Statuses: ok (model matched the truth), corrected (truth
    replaced it), fallback (nothing to check against and the output cannot be trusted), unchecked.
    """
    if truth is None:
        answer = model_answer.strip()
        if looks_degenerate(answer):
            return {"status": "fallback", "final": FALLBACK, "truth": None}
        refusing = any(m in answer.lower() for m in REFUSAL_MARKERS)
        if not refusing:
            if not has_context:
                # no retrieved evidence and no calculator: anything factual here is invented
                return {"status": "fallback", "final": FALLBACK, "truth": None}
            if context and not _supported_by(answer, context):
                # there was a context, but the answer is not actually grounded in it
                return {"status": "fallback", "final": FALLBACK, "truth": None}
        return {"status": "unchecked", "final": answer, "truth": None}
    t = truth["answer"]
    if not model_answer.strip():
        return {"status": "corrected", "final": t, "truth": t}

    tf = _key_facts(t)
    mf = _key_facts(model_answer)
    # numbers and versions are not "close enough" — they must match exactly
    numeric = {x for x in tf if any(c.isdigit() for c in x)}
    if numeric and not numeric <= mf:
        return {"status": "corrected", "final": t, "truth": t}
    values = {x for x in tf if len(x) > 3}
    hit = len(values & mf) / max(1, len(values))
    negated = ("no," in model_answer.lower()) != ("no," in t.lower())
    # the grounded text is always what gets shown; "ok" just records that the model agreed
    return {"status": "ok" if (hit >= 0.75 and not negated) else "corrected", "final": t,
            "truth": t}
