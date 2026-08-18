"""
Pre-tokenize corpus into binary file for fast training
"""
import os
import torch
from tokenizers import Tokenizer
from .tokenizer import TikTokenizerWrapper
from tqdm import tqdm
import sys

def preprocess(corpus_path, tokenizer_path, out_path, max_tokens=None):
    hf = Tokenizer.from_file(tokenizer_path)
    wrapper = TikTokenizerWrapper(hf)

    # stream file
    chunk_size_chars = 2_000_000  # 2M chars per chunk
    all_tokens = []
    total_chars = os.path.getsize(corpus_path)
    print(f"Preprocessing {corpus_path} {total_chars/1e6:.1f}MB chars -> {out_path}")
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(total=total_chars, unit='chars')
        while True:
            chunk = f.read(chunk_size_chars)
            if not chunk:
                break
            ids = wrapper.encode(chunk)
            all_tokens.extend(ids)
            pbar.update(len(chunk))
            # flush print every chunk
            # if max_tokens and we have enough, cut
            if max_tokens and len(all_tokens) >= max_tokens:
                print(f"Reached max_tokens {max_tokens}, truncating")
                all_tokens = all_tokens[:max_tokens]
                break
        pbar.close()

    print(f"Total tokens: {len(all_tokens):,}")
    # save as torch tensor or numpy
    tensor = torch.tensor(all_tokens, dtype=torch.int32)  # vocab < 65535 fits in int32, use int32 to save mem vs int64
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(tensor, out_path)
    print(f"Saved tokens to {out_path}, size {os.path.getsize(out_path)/1024/1024:.1f} MB")
    # also save meta
    meta = {
        'vocab_size': wrapper.vocab_size(),
        'total_tokens': len(all_tokens),
        'chars': total_chars,
        'ratio': total_chars / len(all_tokens) if len(all_tokens) else 0
    }
    import json
    with open(out_path+'.json','w') as jf:
        json.dump(meta,jf,indent=2)
    print(meta)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--out", default="data/tokens.pt")
    ap.add_argument("--max_tokens", type=int, default=20_000_000)
    args = ap.parse_args()
    preprocess(args.corpus, args.tokenizer, args.out, max_tokens=args.max_tokens)
