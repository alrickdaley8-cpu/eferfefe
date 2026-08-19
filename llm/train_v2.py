"""
Upgraded Training v2 for TinyLLM 1M / 20M
Improvements over v1:
- SwiGLU MLP (better per-param capacity)
- Dropout 0.1 for regularization
- Gradient accumulation (effective batch 32768 tokens)
- torch.compile for ~30% speedup
- EMA weights (0.999) for better final checkpoint
- AdamW with no weight decay on norms/biases
- Cosine LR with longer warmup 200 steps
- Validation split 5% + eval every 200 steps
- Improved data loading with random offset augmentation
- Better initialization, label smoothing option
- JSON logging + sample generation with diverse prompts
- Auto-resume from latest checkpoint
"""
import os, math, json, time, glob, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler
from tqdm import tqdm
from tokenizers import Tokenizer

from .config import TinyConfig, TinyConfigV2
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper
from .dataset_fast import FastTokenDataset
from .preprocess import preprocess as preprocess_fn

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr_max * (step+1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))

def get_param_groups(model, weight_decay):
    # Exclude norms and biases from weight decay (common practice)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ln' in name.lower() or 'norm' in name.lower() or 'bias' in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def train_v2():
    cfg = TinyConfigV2()
    print(f"=== TinyLLM v2 Upgraded Training ===")
    print(f"Architecture: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} vocab={cfg.vocab_size} d_ff={cfg.d_ff} swiglu={cfg.use_swiglu} dropout={cfg.dropout}")
    print(f"Params estimate: {cfg.param_count():,}")
    eff_tokens = cfg.effective_tokens_per_batch()
    print(f"Micro batch: {cfg.batch_size} * {cfg.max_seq_len} = {cfg.batch_size*cfg.max_seq_len} tokens")
    print(f"Grad accum {cfg.grad_accum_steps} => Effective batch {eff_tokens} tokens")
    print(f"Steps: micro {cfg.micro_steps} effective {cfg.max_steps} => total {eff_tokens*cfg.max_steps:,} tokens (target {cfg.total_tokens:,})")
    print(f"Upgrades: compile={cfg.use_compile}, ema={cfg.use_ema} ({cfg.ema_decay}), val_split={cfg.val_split}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Check corpus and tokenizer
    if not os.path.exists(cfg.data_path):
        print(f"Corpus missing, generating...")
        from .build_corpus import generate_corpus
        generate_corpus(cfg.data_path, 85_000_000)

    if not os.path.exists(cfg.tokenizer_path):
        from .tokenizer import train_tokenizer
        train_tokenizer(cfg.data_path, cfg.tokenizer_path, vocab_size=cfg.vocab_size)

    hf_tok = Tokenizer.from_file(cfg.tokenizer_path)
    tok_wrap = TikTokenizerWrapper(hf_tok)
    print(f"Tokenizer vocab: {tok_wrap.vocab_size()}")

    # Pre-tokenized
    tokens_pt = "data/tokens.pt"
    if not os.path.exists(tokens_pt):
        print("Pre-tokenizing 20M tokens...")
        preprocess_fn(cfg.data_path, cfg.tokenizer_path, tokens_pt, max_tokens=cfg.total_tokens)
    else:
        print(f"Using {tokens_pt}")

    # Dataset with train/val split
    full_dataset = FastTokenDataset(tokens_pt, seq_len=cfg.max_seq_len)
    n_total = len(full_dataset)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val
    print(f"Dataset: total {n_total} seqs, train {n_train}, val {n_val}")

    train_dataset = Subset(full_dataset, range(n_train))
    val_dataset = Subset(full_dataset, range(n_train, n_total))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    # Try resume latest V2 only (to avoid arch mismatch with V1 GELU)
    latest_ckpt = None
    candidates = glob.glob(os.path.join(cfg.checkpoint_dir, "model_v2*.pt"))
    if candidates:
        latest_ckpt = max(candidates, key=lambda x: os.path.getmtime(x))
        print(f"Resuming from {latest_ckpt}")

    if latest_ckpt and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        try:
            # Ensure config is V2 compatible
            ckpt_cfg = ckpt.get('config', cfg)
            if getattr(ckpt_cfg, 'use_swiglu', False) != cfg.use_swiglu:
                print(f"Config mismatch (swiglu), starting fresh")
                raise ValueError("swiglu mismatch")
            model = TinyLLM(cfg).to(device)
            model.load_state_dict(ckpt['model'])
            start_step = ckpt.get('step', 0)
            print(f"Resumed step {start_step}")
        except Exception as e:
            print(f"Resume failed {e}, starting from scratch with v2 config")
            model = TinyLLM(cfg).to(device)
            start_step = 0
    else:
        print("No V2 checkpoint found, starting from scratch")
        model = TinyLLM(cfg).to(device)
        start_step = 0

    actual_params = model.count_params()
    print(f"Actual params: {actual_params:,}")

    # Compile
    if cfg.use_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
            print("Compile OK")
        except Exception as e:
            print(f"Compile failed: {e}")

    # Optimizer with no decay for norms
    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr_max, betas=cfg.betas)

    # EMA
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    # For logging
    log_file = os.path.join(cfg.checkpoint_dir, "train_v2_log.jsonl")
    best_val_loss = float('inf')

    model.train()
    step = start_step
    total_loss = 0
    micro_step = 0
    optimizer.zero_grad()

    # Diverse prompts for eval samples
    sample_prompts = [
        "Once upon a time, Lily",
        "Q: What is 12 + 8? A:",
        "Explain why the sky is blue",
        "Write a Python function to add",
        "User: Hello! Who are you?\nAssistant:",
    ]

    pbar = tqdm(total=cfg.micro_steps, initial=step, desc="Training v2")
    data_iter = iter(train_loader)
    start_time = time.time()

    while step < cfg.micro_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        # Forward with autocast if CUDA
        if cfg.use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
        else:
            logits, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps

        loss.backward()
        micro_step += 1

        # Gradient accumulation
        if micro_step % cfg.grad_accum_steps == 0:
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema:
                ema.update()
            optimizer.zero_grad()

            # Logging
            # Since we divided loss by accum, multiply back for display
            real_loss = loss.item() * cfg.grad_accum_steps
            total_loss += real_loss
            avg_loss = total_loss / (step+1)

            pbar.set_postfix({
                "loss": f"{real_loss:.3f}",
                "avg": f"{avg_loss:.3f}",
                "lr": f"{lr:.2e}",
                "tok": f"{(step+1)*eff_tokens/1e6:.2f}M",
                "ppl": f"{math.exp(min(real_loss,10)):.1f}"
            })
            pbar.update(1)
            step += 1

            # Log to file
            log_entry = {"step": step, "loss": real_loss, "avg_loss": avg_loss, "lr": lr, "tokens": (step)*eff_tokens, "time": time.time()-start_time}
            with open(log_file, 'a') as lf:
                lf.write(json.dumps(log_entry)+"\n")

            # Validation every 200 steps
            if step % 200 == 0:
                model.eval()
                val_loss = 0
                val_steps = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(device); vy = vy.to(device)
                        _, vloss = model(vx, vy)
                        val_loss += vloss.item()
                        val_steps += 1
                        if val_steps >= 20:  # limit val for speed
                            break
                val_loss /= max(1,val_steps)
                val_ppl = math.exp(min(val_loss,10))
                print(f"\n[Val] step {step} loss {val_loss:.4f} ppl {val_ppl:.2f}")

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(cfg.checkpoint_dir, f"model_v2_best.pt")
                    # Save EMA if available
                    if ema:
                        ema.apply_shadow()
                    torch.save({'model': model.state_dict() if not hasattr(model,'_orig_mod') else model._orig_mod.state_dict(), 'config': cfg, 'step': step, 'loss': avg_loss, 'val_loss': val_loss}, best_path)
                    print(f"New best model saved to {best_path}")
                    if ema:
                        ema.restore()

                # Sample generation
                print(f"\n--- Samples step {step} ---")
                model.eval()
                for prompt in sample_prompts[:3]:
                    try:
                        prompt_ids = tok_wrap.encode(prompt)
                        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            out = model.generate(prompt_tensor, max_new_tokens=60, temperature=0.8, top_k=50)
                        decoded = tok_wrap.decode(out[0].tolist())
                        # show only new part
                        print(f"Prompt: {prompt} -> {decoded[-120:]}")
                    except Exception as e:
                        print(f"Sample failed {e}")
                print("--- End samples ---\n")
                model.train()

            # Checkpoint every 200
            if step % 200 == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_v2_step{step}.pt")
                # Save EMA version as final if enabled
                save_model = model
                if ema:
                    ema.apply_shadow()
                torch.save({
                    'model': save_model.state_dict() if not hasattr(save_model,'_orig_mod') else save_model._orig_mod.state_dict(),
                    'config': cfg,
                    'step': step,
                    'loss': total_loss/(step+1),
                    'val_loss': best_val_loss,
                }, ckpt_path)
                print(f"Checkpoint {ckpt_path}")
                if ema:
                    ema.restore()

            # Early check for full token budget
            if step * eff_tokens >= cfg.total_tokens:
                print(f"Reached token budget {cfg.total_tokens}")
                break

    pbar.close()

    # Final save EMA
    final_path = os.path.join(cfg.checkpoint_dir, "model_v2_final.pt")
    if ema:
        ema.apply_shadow()
    torch.save({
        'model': model.state_dict() if not hasattr(model,'_orig_mod') else model._orig_mod.state_dict(),
        'config': cfg,
        'step': step,
        'loss': total_loss/max(1,step)
    }, final_path)
    if ema:
        ema.restore()
    print(f"Training v2 done! Final saved to {final_path}")
    print(f"Total tokens: {step*eff_tokens:,}")

if __name__ == "__main__":
    train_v2()
