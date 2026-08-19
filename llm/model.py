"""
1M-parameter decoder-only Transformer with RoPE
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TinyConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # RMSNorm: x / sqrt(mean(x^2)+eps) * weight
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight

def build_rope_cache(seq_len, head_dim, device, base=10000):
    # RoPE cache: [seq_len, head_dim//2, 2] -> cos, sin
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device)
    idx_theta = torch.outer(seq_idx, theta) # [seq_len, head_dim//2]
    cos = idx_theta.cos()
    sin = idx_theta.sin()
    return cos, sin

def apply_rope(q, k, cos, sin):
    # q,k: [B, n_heads, T, head_dim]
    # cos,sin: [T, head_dim//2]
    # rotate half
    def rotate_half(x):
        # x: [..., head_dim]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # (x1, x2) -> (-x2, x1)
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    # cos,sin need broadcasting: [1,1,T,head_dim//2] but we have interleaved
    # Expand cos/sin to head_dim: repeat each
    T = q.size(2)
    cos = cos[:T] # [T, Hd//2]
    sin = sin[:T]
    # reshape for broadcast: [1,1,T, Hd//2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # split even/odd
    q1 = q[..., ::2]
    q2 = q[..., 1::2]
    k1 = k[..., ::2]
    k2 = k[..., 1::2]

    q_rot = torch.stack((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1).flatten(-2)
    k_rot = torch.stack((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1).flatten(-2)
    return q_rot, k_rot

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        qkv = self.qkv(x) # B,T,3*C
        q, k, v = qkv.split(C, dim=-1)
        # reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2) # B,nh,T,Hd
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        q, k = apply_rope(q, k, cos, sin)

        # causal attention
        # use scaled_dot_product_attention if available (PyTorch 2.0+)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        attn_out = attn_out.transpose(1,2).contiguous().view(B,T,C)
        return self.out(attn_out)

class MLP(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.use_swiglu = getattr(cfg, 'use_swiglu', False)
        if self.use_swiglu:
            # SwiGLU: gate, up, down — keeps params similar if d_ff=256 vs 384 GELU (3*96*256=73k)
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
    def forward(self, x):
        if self.use_swiglu:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        else:
            return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout>0 else nn.Identity()
    def forward(self, x, cos, sin):
        x = x + self.dropout(self.attn(self.ln1(x), cos, sin))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class TinyLLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # init
        self.apply(self._init_weights)

        # RoPE cache
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads, device='cpu')
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: [B,T]
        B, T = idx.size()
        assert T <= self.cfg.max_seq_len, f"seq len {T} > max {self.cfg.max_seq_len}"
        x = self.tok_emb(idx) # B,T,C

        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B,T,V

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9):
        # idx: [B,T] prompt
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)

            # top-p
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                # zero out
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # sample from sorted
                next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = torch.gather(sorted_idx, -1, next_token_sorted)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
        return idx
