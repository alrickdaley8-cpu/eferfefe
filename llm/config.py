"""
Tiny 1M-parameter LLM Config
Target: ~1.06M parameters with 20M tokens (Chinchilla optimal 20 tok/param)
"""
from dataclasses import dataclass

@dataclass
class TinyConfig:
    # architecture: tuned to ~1M params
    vocab_size: int = 4096        # BPE vocab
    max_seq_len: int = 256        # context length
    d_model: int = 96             # hidden dim
    n_layers: int = 6             # depth helps reasoning
    n_heads: int = 6              # 96/6=16 per head
    d_ff: int = 384               # 4*d_model
    dropout: float = 0.0          # no dropout for tiny data
    tie_weights: bool = True      # tie token emb and lm_head

    # training: 20M token budget
    total_tokens: int = 20_000_000
    batch_size: int = 32          # sequences per batch
    # tokens per batch = batch_size * max_seq_len = 8192
    # steps for 20M = 20M / 8192 ≈ 2442
    max_steps: int = 2442
    warmup_steps: int = 100
    lr_max: float = 5e-4
    lr_min: float = 5e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # io
    data_path: str = "data/corpus.txt"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    checkpoint_dir: str = "checkpoints"

    def tokens_per_batch(self):
        return self.batch_size * self.max_seq_len

    def param_count(self):
        # rough estimate (no bias)
        tok_emb = self.vocab_size * self.d_model
        # RoPE: no pos emb
        per_layer = (
            3 * self.d_model * self.d_model +  # qkv
            self.d_model * self.d_model +      # out
            self.d_model * self.d_ff +         # ff1
            self.d_ff * self.d_model           # ff2
        )
        # layernorms: minimal
        norm_params = self.n_layers * 2 * self.d_model + self.d_model
        total = tok_emb + self.n_layers * per_layer + norm_params
        if not self.tie_weights:
            total += self.vocab_size * self.d_model
        return total

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
