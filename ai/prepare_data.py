"""Tokenize the raw corpus into flat uint16 token streams (train.bin / val.bin)."""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ai", "data")
CORPUS = os.path.join(DATA, "corpus")
EOT = "<|endoftext|>"


def doc_iter(path: str):
    """Yield documents from a shard (documents are separated by the EOT marker line)."""
    buf: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                doc = "".join(buf).strip()
                buf = []
                if doc:
                    yield doc
            else:
                buf.append(line)
    if buf and "".join(buf).strip():
        yield "".join(buf).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=100_000_000, help="hard cap on train tokens")
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=1000, help="documents per encode_batch call")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    eot_id = tok.token_to_id(EOT)
    shards = sorted(glob.glob(os.path.join(CORPUS, "shard_*.txt")))
    assert shards, "no corpus shards"
    assert tok.get_vocab_size() <= 65536

    val_path, train_path = os.path.join(DATA, "val.bin"), os.path.join(DATA, "train.bin")
    fval, ftrain = open(val_path, "wb"), open(train_path, "wb")
    n_val = n_train = 0

    def flush(ids: list[int]) -> bool:
        nonlocal n_val, n_train
        arr = np.asarray(ids, dtype=np.uint16)
        if n_val < args.val_tokens:
            take = min(len(arr), args.val_tokens - n_val)
            arr[:take].tofile(fval)
            n_val += take
            arr = arr[take:]
        if len(arr):
            take = min(len(arr), args.max_tokens - n_train)
            arr[:take].tofile(ftrain)
            n_train += take
        return n_train >= args.max_tokens

    done = False
    for si, shard in enumerate(shards):
        if done:
            break
        batch: list[str] = []
        for doc in doc_iter(shard):
            batch.append(doc)
            if len(batch) >= args.batch:
                ids: list[int] = []
                for enc in tok.encode_batch(batch):
                    ids.extend(enc.ids)
                    ids.append(eot_id)
                batch = []
                if flush(ids):
                    done = True
                    break
        if batch and not done:
            ids = []
            for enc in tok.encode_batch(batch):
                ids.extend(enc.ids)
                ids.append(eot_id)
            done = flush(ids)
        print(f"[prepare] shard {si+1}/{len(shards)}  train={n_train/1e6:.2f}M  val={n_val/1e6:.2f}M",
              flush=True)

    fval.close()
    ftrain.close()
    print(f"[prepare] DONE train={n_train:,} tokens -> {train_path}")
    print(f"[prepare] DONE val  ={n_val:,} tokens -> {val_path}")


if __name__ == "__main__":
    main()
