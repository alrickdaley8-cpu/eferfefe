"""
Fast dataset from pre-tokenized .pt file
"""
import torch
from torch.utils.data import Dataset
import os

class FastTokenDataset(Dataset):
    def __init__(self, tokens_path, seq_len=256):
        print(f"Loading tokens from {tokens_path}")
        data = torch.load(tokens_path, map_location='cpu')
        # data is int32
        self.data = data.long()  # convert to long for embedding
        self.seq_len = seq_len
        self.n_seq = (len(self.data) - 1) // seq_len
        print(f"Tokens: {len(self.data):,} -> sequences: {self.n_seq:,} (seq_len {seq_len}) => usable tokens {self.n_seq*seq_len:,}")

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # we need seq_len+1 for target? Actually we can take seq_len and shift
        x = self.data[start:start+self.seq_len]
        y = self.data[start+1:start+self.seq_len+1]
        return x, y
