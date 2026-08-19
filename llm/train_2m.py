"""
Training 2M model - as powerful as possible within 2M budget
- 2,017,152 params (128 dim, 7 layers, 8 heads, 384 ff SwiGLU, vocab 4096, 512 context)
- 40M tokens Chinchilla optimal (20*params)
- SwiGLU, RMSNorm, RoPE, dropout 0.1, tied weights
- Grad accum 4x effective batch 65536 tokens
- EMA 0.999, validation 5%, best tracking
- Auto-resume, JSON logging, sample generation
"""
import os, math, json, time, glob
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tokenizers import Tokenizer

from .config import TinyConfig2M
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

def get_param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'ln' in name.lower() or 'norm' in name.lower() or 'bias' in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1-self.decay)*p.data + self.decay*self.shadow[n]
    def apply(self, model):
        self.backup = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[n]
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]

def train_2m():
    cfg = TinyConfig2M()
    print(f"=== TinyLLM 2M Powerful Training ===")
    print(f"Architecture: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} d_ff={cfg.d_ff} vocab={cfg.vocab_size} seq={cfg.max_seq_len} swiglu={cfg.use_swiglu}")
    print(f"Params: {cfg.param_count():,} actual")
    eff = cfg.effective_tokens_per_batch()
    print(f"Micro batch {cfg.batch_size}*{cfg.max_seq_len}={cfg.batch_size*cfg.max_seq_len} tokens, grad_accum {cfg.grad_accum_steps} => eff {eff} tokens")
    print(f"Steps: micro {cfg.micro_steps} effective {cfg.max_steps} => total {eff*cfg.max_steps:,} tokens (target {cfg.total_tokens:,})")
    print(f"Upgrades: SwiGLU, dropout {cfg.dropout}, EMA {cfg.ema_decay}, val {cfg.val_split}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Tokenizer
    if not os.path.exists(cfg.tokenizer_path):
        from .tokenizer import train_tokenizer
        train_tokenizer(cfg.data_path, cfg.tokenizer_path, vocab_size=cfg.vocab_size)

    hf = Tokenizer.from_file(cfg.tokenizer_path)
    tok_wrap = TikTokenizerWrapper(hf)
    print(f"Tokenizer vocab: {tok_wrap.vocab_size()}")

    # Tokens: need 40M for 2M Chinchilla
    tokens_40m = "data/tokens_40m.pt"
    tokens_20m = "data/tokens.pt"
    target_tokens_path = tokens_40m if os.path.exists(tokens_40m) else tokens_20m

    if not os.path.exists(tokens_40m):
        print(f"40M tokens not found, checking 20M and generating additional if needed...")
        if not os.path.exists(tokens_20m):
            print("No tokens found, preprocessing 20M...")
            preprocess_fn(cfg.data_path, cfg.tokenizer_path, tokens_20m, max_tokens=20_000_000)
        # Try to create 40M by using corpus + fresh
        if os.path.getsize(tokens_20m) < 150_000_000:  # 20M tokens ~77MB, 40M ~154MB
            print("Generating additional 20M tokens for 40M total...")
            # Generate more corpus if needed
            if not os.path.exists("data/corpus.txt") or os.path.getsize("data/corpus.txt") < 150_000_000:
                from .build_corpus import generate_corpus
                print("Generating larger corpus 160M chars for 40M tokens...")
                generate_corpus("data/corpus_40m.txt", target_chars=160_000_000)
                preprocess_fn("data/corpus_40m.txt", cfg.tokenizer_path, tokens_40m, max_tokens=40_000_000)
                target_tokens_path = tokens_40m
            else:
                # Use existing corpus but slice 40M
                print("Using existing corpus to generate 40M tokens...")
                preprocess_fn(cfg.data_path, cfg.tokenizer_path, tokens_40m, max_tokens=40_000_000)
                target_tokens_path = tokens_40m

    print(f"Using tokens: {target_tokens_path}")

    full_ds = FastTokenDataset(target_tokens_path, seq_len=cfg.max_seq_len)
    n_val = int(len(full_ds)*cfg.val_split)
    n_train = len(full_ds)-n_val
    print(f"Dataset: total {len(full_ds)} seqs, train {n_train}, val {n_val}")

    train_ds = Subset(full_ds, range(n_train))
    val_ds = Subset(full_ds, range(n_train, len(full_ds)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resume
    latest = None
    cands = glob.glob(os.path.join(cfg.checkpoint_dir, "model_2m_*.pt"))
    if cands:
        latest = max(cands, key=lambda x: os.path.getmtime(x))
        print(f"Resuming from {latest}")

    if latest and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model = TinyLLM(cfg).to(device)
        try:
            model.load_state_dict(ckpt['model'])
            start_step = ckpt.get('step',0)
            print(f"Resumed step {start_step}")
        except Exception as e:
            print(f"Resume failed {e}, fresh start")
            model = TinyLLM(cfg).to(device)
            start_step = 0
    else:
        model = TinyLLM(cfg).to(device)
        start_step = 0

    print(f"Actual params: {model.count_params():,}")

    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr_max, betas=cfg.betas)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    best_val = float('inf')
    model.train()
    step = start_step
    total_loss = 0
    micro = 0
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    pbar = tqdm(total=cfg.micro_steps, initial=step, desc="Training 2M")
    start_time = time.time()
    log_file = os.path.join(cfg.checkpoint_dir, "train_2m_log.jsonl")

    sample_prompts = [
        "What is the capital of France? Explain in depth.",
        "What is photosynthesis? Explain step by step.",
        "Explain AI in depth.",
        "Write a Python function to add two numbers.",
        "Once upon a time, Lily found a magic key.",
    ]

    while step < cfg.micro_steps:
        try:
            x,y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x,y = next(data_iter)

        x=x.to(device); y=y.to(device)
        logits, loss = model(x,y)
        (loss/cfg.grad_accum_steps).backward()
        micro+=1

        if micro % cfg.grad_accum_steps == 0:
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg['lr']=lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema:
                ema.update(model)
            optimizer.zero_grad()

            real_loss = loss.item()
            total_loss+=real_loss
            avg = total_loss/(step+1)
            pbar.set_postfix({"loss": f"{real_loss:.3f}", "avg": f"{avg:.3f}", "lr": f"{lr:.2e}", "tok": f"{(step+1)*eff/1e6:.1f}M"})
            pbar.update(1)
            step+=1

            with open(log_file,'a') as f:
                f.write(json.dumps({"step":step,"loss":real_loss,"avg":avg,"lr":lr,"tokens_m":step*eff/1e6})+"\n")

            if step % 200 == 0:
                # val
                model.eval()
                vloss=0; cnt=0
                with torch.no_grad():
                    for vx,vy in val_loader:
                        vx=vx.to(device); vy=vy.to(device)
                        _, vl = model(vx,vy)
                        vloss+=vl.item()
                        cnt+=1
                        if cnt>=20:
                            break
                vloss/=max(1,cnt)
                print(f"\n[Val] step {step} val_loss {vloss:.4f} ppl {pow(2.718, min(vloss,10)):.2f}")

                if vloss < best_val:
                    best_val=vloss
                    if ema:
                        ema.apply(model)
                    torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'loss': avg, 'val_loss': vloss}, os.path.join(cfg.checkpoint_dir, "model_2m_best.pt"))
                    print(f"New best saved val {vloss:.4f}")
                    if ema:
                        ema.restore(model)

                # samples
                print(f"\n--- Samples step {step} ---")
                for prompt in sample_prompts[:2]:
                    try:
                        ids = tok_wrap.encode(prompt)
                        t = torch.tensor([ids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            out = model.generate(t, max_new_tokens=80, temperature=0.7, top_k=50)
                        print(f"Prompt: {prompt}\nGen: {tok_wrap.decode(out[0].tolist())[-150:]}\n")
                    except Exception as e:
                        print(f"Sample fail {e}")
                print("--- End samples ---\n")
                model.train()

                # checkpoint
                if ema:
                    ema.apply(model)
                torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'loss': avg}, os.path.join(cfg.checkpoint_dir, f"model_2m_step{step}.pt"))
                if ema:
                    ema.restore(model)

    pbar.close()
    if ema:
        ema.apply(model)
    torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'loss': total_loss/max(1,step)}, os.path.join(cfg.checkpoint_dir, "model_2m_final.pt"))
    print(f"Training 2M done! Final saved")

if __name__ == "__main__":
    train_2m()
