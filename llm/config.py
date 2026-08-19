"""
Tiny 1M-parameter LLM Config
Target: ~1.06M parameters with 20M tokens (Chinchilla optimal 20 tok/param)
Upgraded v2 includes SwiGLU, better training, EMA, compile, etc.
"""
from dataclasses import dataclass, field

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
    use_swiglu: bool = False      # v1 default GELU, v2 SwiGLU

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
        if self.use_swiglu:
            # SwiGLU has 3 matrices in MLP
            per_layer_mlp = 3 * self.d_model * self.d_ff
        else:
            per_layer_mlp = 2 * self.d_model * self.d_ff
        per_layer = (
            3 * self.d_model * self.d_model +  # qkv
            self.d_model * self.d_model +      # out
            per_layer_mlp
        )
        # layernorms: minimal
        norm_params = self.n_layers * 2 * self.d_model + self.d_model
        total = tok_emb + self.n_layers * per_layer + norm_params
        if not self.tie_weights:
            total += self.vocab_size * self.d_model
        return total

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@dataclass
class TinyConfigV2(TinyConfig):
    """Upgraded training config - faster, better, more stable"""
    # Architecture upgrades
    use_swiglu: bool = True
    d_ff: int = 256               # smaller for SwiGLU to keep ~1M params: 3*96*256=73k same as 2*96*384
    dropout: float = 0.1          # add dropout for regularization

    # Training upgrades
    batch_size: int = 32
    grad_accum_steps: int = 4     # effective batch 128 => 32768 tokens
    max_steps: int = 610          # 20M / 32768 ≈ 610 effective steps, but we still do micro steps 2440
    micro_steps: int = 2442       # actual optimizer steps unchanged, but for v2 we use effective counting
    warmup_steps: int = 200       # longer warmup
    lr_max: float = 6e-4          # slightly higher for SwiGLU
    lr_min: float = 6e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Advanced
    use_compile: bool = False     # disabled on CPU env (needs gcc+python-dev), auto-enable for CUDA
    use_ema: bool = True
    ema_decay: float = 0.999
    use_amp: bool = False         # CPU doesn't benefit, but auto for CUDA
    label_smoothing: float = 0.0

    # Data
    val_split: float = 0.05
    use_mixed_data: bool = True   # better mixing with chat data

    def effective_tokens_per_batch(self):
        return self.batch_size * self.grad_accum_steps * self.max_seq_len
