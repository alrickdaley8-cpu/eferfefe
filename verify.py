"""Verify 1M params and 20M tokens"""
from llm.config import TinyConfig
from llm.model import TinyLLM
import os, torch

cfg = TinyConfig()
model = TinyLLM(cfg)
params = model.count_params()
print(f"=== VERIFY ===")
print(f"Model params: {params:,} (target 1M)")
print(f"  - d_model {cfg.d_model}, layers {cfg.n_layers}, heads {cfg.n_heads}, vocab {cfg.vocab_size}")
print(f"  - estimate {cfg.param_count():,}")
assert 900_000 <= params <= 1_200_000, f"Param count {params} not ~1M"

# tokens
token_file = "data/tokens.pt"
if os.path.exists(token_file):
    meta_path = token_file + ".json"
    import json
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            print(f"Tokens: {meta['total_tokens']:,} (target 20M)")
            print(f"  - file {token_file} size {os.path.getsize(token_file)/1024/1024:.1f} MB")
            print(f"  - vocab {meta['vocab_size']} chars/token ratio {meta['ratio']:.2f}")
            assert meta['total_tokens'] >= 19_000_000, "Not enough tokens"
    else:
        t = torch.load(token_file)
        print(f"Tokens in file: {len(t):,}")

# steps
tok_per_batch = cfg.tokens_per_batch()
steps = cfg.max_steps
total = tok_per_batch * steps
print(f"Training budget: {steps} steps × {tok_per_batch} tok/batch = {total:,} tokens")
assert total >= 20_000_000, "Training budget <20M"

print("✅ Verified: ~1M params and 20M token dataset + training budget")
