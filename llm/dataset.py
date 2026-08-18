"""
Dataset handling for 20M token budget
"""
import os
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, corpus_path, tokenizer_wrapper, seq_len=256):
        self.seq_len = seq_len
        self.tok = tokenizer_wrapper

        print(f"Loading corpus {corpus_path} ...")
        with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        print(f"Raw chars: {len(text):,}")
        # encode all - can be large, so chunk
        # For 80M chars, this might be heavy (~20M tokens). Do streaming chunk encode.
        chunk_size = 10_000_000 # 10M chars per chunk
        all_ids = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            ids = self.tok.encode(chunk)
            all_ids.extend(ids)
            print(f"  encoded chunk {i//chunk_size+1}/{(len(text)+chunk_size-1)//chunk_size}: total tokens {len(all_ids):,}")

        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"Total tokens in corpus: {len(self.data):,}")

        # Trim to multiple of seq_len?
        self.n_sequences = len(self.data) // (seq_len+1)
        print(f"Total sequences ({seq_len}+1): {self.n_sequences:,}")
        print(f"Tokens effectively used for training: {self.n_sequences * seq_len:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * (self.seq_len+1)
        chunk = self.data[start:start+self.seq_len+1]
        if len(chunk) < self.seq_len+1:
            # pad
            pad = torch.full((self.seq_len+1 - len(chunk),), self.tok.pad_id, dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def count_tokens(corpus_path, tokenizer_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    from .tokenizer import TikTokenizerWrapper
    wrapper = TikTokenizerWrapper(tok)
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read(5_000_000) # sample 5MB
        ids = wrapper.encode(text)
        print(f"Sample 5M chars => {len(ids)} tokens, ratio {len(text)/len(ids):.2f} chars/token")
        # extrapolate
        full_chars = os.path.getsize(corpus_path)
        est_tokens = int(full_chars / len(text) * len(ids))
        print(f"Full file {full_chars/1e6:.2f} MB chars => est {est_tokens:,} tokens")
