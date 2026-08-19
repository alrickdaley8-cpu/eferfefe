"""Train a byte-level BPE tokenizer (vocab 8192) on the corpus."""
from __future__ import annotations

import argparse
import glob
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "ai", "data", "corpus")
OUT = os.path.join(ROOT, "ai", "data", "tokenizer.json")

EOT = "<|endoftext|>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-shards", type=int, default=6, help="shards to sample for training")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(CORPUS, "shard_*.txt")))
    assert files, "no corpus shards; run ai/build_corpus.py first"
    step = max(1, len(files) // args.max_shards)
    files = files[::step][: args.max_shards]
    print(f"training tokenizer on {len(files)} shards")

    tok = Tokenizer(models.BPE(unk_token=None))
    # GPT-2 style byte-level pre-tokenization (regex splitting included).
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train(files, trainer)
    tok.save(OUT)
    print(f"saved {OUT}  vocab={tok.get_vocab_size()}  eot_id={tok.token_to_id(EOT)}")


if __name__ == "__main__":
    main()
