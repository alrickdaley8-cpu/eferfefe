"""Build a raw text corpus (~400 MB -> ~100M BPE tokens) from PyPI source releases.

Why PyPI: it is the only bulk text source reachable from this sandbox, and it gives a
healthy mix of English prose (READMEs, docs, docstrings) and Python code.

Usage:
    python ai/build_corpus.py --target-mb 420 --workers 24
Output:
    ai/data/corpus/shard_XXX.txt   (documents separated by a blank line + <|endoftext|>)
    ai/data/corpus/manifest.jsonl  (one record per package that contributed text)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "ai", "data", "corpus")
TOP_CSV = os.path.join(ROOT, "ai", "data", "top-pypi-packages.csv")

TEXT_EXT = (".py", ".md", ".rst", ".txt", ".cfg", ".toml", ".ini")
SKIP_PAT = re.compile(
    r"(site-packages|/tests?/|/test_|_test\.py|/vendor/|/third_party/|\.min\.|LICENSE|COPYING)", re.I
)
EOT = "<|endoftext|>"

_lock = threading.Lock()
_state = {"bytes": 0, "docs": 0, "pkgs": 0, "shard": 0, "fh": None, "t0": time.time()}


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def is_texty(name: str) -> bool:
    return name.lower().endswith(TEXT_EXT) and not SKIP_PAT.search(name)


def clean(text: str) -> str | None:
    """Light quality filter; drop binary-ish / tiny / repetitive blobs."""
    if len(text) < 200:
        return None
    if text.count("\x00"):
        return None
    # printable ratio
    printable = sum(c.isprintable() or c in "\n\t" for c in text[:4000])
    if printable / min(len(text), 4000) < 0.95:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    if max(len(l) for l in lines) > 2000:  # generated / base64 blobs
        return None
    uniq = len(set(lines)) / len(lines)
    if uniq < 0.3:
        return None
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"


def open_shard() -> None:
    if _state["fh"] is not None:
        _state["fh"].close()
    path = os.path.join(OUT_DIR, f"shard_{_state['shard']:03d}.txt")
    _state["fh"] = open(path, "w", encoding="utf-8")
    _state["shard"] += 1


SHARD_BYTES = 32 * 1024 * 1024


def emit(docs: list[str], pkg: str) -> None:
    with _lock:
        fh = _state["fh"]
        for d in docs:
            fh.write(d)
            fh.write("\n" + EOT + "\n")
            _state["bytes"] += len(d) + len(EOT) + 2
            _state["docs"] += 1
        _state["pkgs"] += 1
        if fh.tell() > SHARD_BYTES:
            open_shard()
        if _state["pkgs"] % 100 == 0:
            mb = _state["bytes"] / 1e6
            el = time.time() - _state["t0"]
            log(f"[corpus] {mb:8.1f} MB  docs={_state['docs']:7d}  pkgs={_state['pkgs']:6d}  "
                f"{mb/max(el,1):.2f} MB/s  last={pkg}")


def files_from_archive(content: bytes, url: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        if url.endswith((".tar.gz", ".tgz", ".tar.bz2")):
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile() or m.size > 512_000 or not is_texty(m.name):
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    out.append((m.name, f.read().decode("utf-8", "ignore")))
        elif url.endswith((".zip", ".whl")):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for m in zf.infolist():
                    if m.file_size > 512_000 or not is_texty(m.filename):
                        continue
                    out.append((m.filename, zf.read(m).decode("utf-8", "ignore")))
    except Exception:
        return []
    return out


def pick_url(info: dict) -> str | None:
    urls = info.get("urls") or []
    sdists = [u for u in urls if u["packagetype"] == "sdist"]
    wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
    for cand in (sdists, wheels):
        if cand:
            u = min(cand, key=lambda x: x["size"])
            if 2_000 < u["size"] < 40_000_000:
                return u["url"]
    return None


def handle(pkg: str, sess: requests.Session, seen: set[str], manifest) -> None:
    if _state["bytes"] >= _state["limit"]:
        return
    try:
        r = sess.get(f"https://pypi.org/pypi/{pkg}/json", timeout=20)
        if r.status_code != 200:
            return
        info = r.json()
        url = pick_url(info)
        if not url:
            return
        blob = sess.get(url, timeout=60).content
    except Exception:
        return

    docs, nbytes = [], 0
    for name, text in files_from_archive(blob, url):
        c = clean(text)
        if c is None:
            continue
        h = hashlib.md5(c.encode()).hexdigest()
        with _lock:
            if h in seen:
                continue
            seen.add(h)
        docs.append(f"# file: {os.path.basename(name)} (package: {pkg})\n{c}")
        nbytes += len(c)
        if nbytes > 1_200_000:  # cap a single package's contribution
            break
    if docs:
        emit(docs, pkg)
        with _lock:
            manifest.write(json.dumps({"package": pkg, "docs": len(docs), "bytes": nbytes}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=420.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-packages", type=int, default=15000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    _state["limit"] = args.target_mb * 1e6
    open_shard()

    with open(TOP_CSV) as f:
        pkgs = [row["project"] for row in csv.DictReader(f)][: args.max_packages]
    random.Random(0).shuffle(pkgs)  # mix popular/less popular through the run
    log(f"[corpus] {len(pkgs)} packages, target {args.target_mb} MB -> {OUT_DIR}")

    seen: set[str] = set()
    sess = requests.Session()
    sess.headers["User-Agent"] = "tiny-lm-corpus-builder/1.0"
    with open(os.path.join(OUT_DIR, "manifest.jsonl"), "w") as manifest:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(handle, p, sess, seen, manifest) for p in pkgs]
            for fu in futs:
                fu.result()
                if _state["bytes"] >= _state["limit"]:
                    break
            ex.shutdown(wait=False, cancel_futures=True)
    if _state["fh"]:
        _state["fh"].close()
    log(f"[corpus] DONE {_state['bytes']/1e6:.1f} MB, {_state['docs']} docs, {_state['pkgs']} packages")


if __name__ == "__main__":
    main()
