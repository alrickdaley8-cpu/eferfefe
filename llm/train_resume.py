"""
Resume training from latest v1 checkpoint with upgraded training loop
- Keeps original architecture (GELU, d_ff 384) for compatibility
- Adds: EMA, grad accum, validation, best tracking, auto-resume, JSON logging
"""
import os, math, json, time, glob
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tokenizers import Tokenizer

from .config import TinyConfig
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper
from .dataset_fast import FastTokenDataset

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

def train_resume():
    cfg = TinyConfig()
    cfg.warmup_steps = 200
    cfg.batch_size = 32
    grad_accum = 4
    eff_batch_tokens = cfg.batch_size * grad_accum * cfg.max_seq_len
    print(f"=== Resume Upgraded Training v1 ===")
    print(f"Effective batch: {cfg.batch_size}*{grad_accum}*{cfg.max_seq_len} = {eff_batch_tokens} tokens")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    hf = Tokenizer.from_file(cfg.tokenizer_path)
    tok = TikTokenizerWrapper(hf)
    tokens_pt = "data/tokens.pt"
    full_ds = FastTokenDataset(tokens_pt, seq_len=cfg.max_seq_len)
    n_val = int(len(full_ds)*0.05)
    n_train = len(full_ds)-n_val
    train_ds = Subset(full_ds, range(n_train))
    val_ds = Subset(full_ds, range(n_train, len(full_ds)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # find latest
    candidates = glob.glob(os.path.join(cfg.checkpoint_dir, "model_step*.pt"))
    if not candidates:
        print("No checkpoint to resume")
        return
    latest = max(candidates, key=lambda x: os.path.getmtime(x))
    print(f"Resuming from {latest}")
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    start_step = ckpt.get('step',0)
    print(f"Start step {start_step}, loss {ckpt.get('loss')}")

    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr_max, betas=cfg.betas)
    ema = EMA(model, decay=0.999)
    best_val = float('inf')

    model.train()
    step = start_step
    total_loss = ckpt.get('loss',0)*start_step if start_step>0 else 0
    micro = 0
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, initial=step)
    log_file = os.path.join(cfg.checkpoint_dir, "train_resume_log.jsonl")

    while step < cfg.max_steps:
        try:
            x,y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x,y = next(data_iter)
        x=x.to(device); y=y.to(device)
        logits, loss = model(x,y)
        (loss/grad_accum).backward()
        micro+=1
        if micro % grad_accum == 0:
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg['lr']=lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)
            optimizer.zero_grad()
            real_loss = loss.item()
            total_loss+=real_loss
            avg = total_loss/(step+1)
            pbar.set_postfix({"loss": f"{real_loss:.3f}", "avg": f"{avg:.3f}", "lr": f"{lr:.2e}", "tok": f"{(step+1)*cfg.tokens_per_batch()*grad_accum/1e6:.2f}M"})
            pbar.update(1)
            step+=1
            with open(log_file,'a') as f:
                f.write(json.dumps({"step":step,"loss":real_loss,"avg":avg,"lr":lr})+"\n")
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
                        if cnt>=20: break
                vloss/=max(1,cnt)
                print(f"\n[Val] step {step} val_loss {vloss:.4f} ppl {math.exp(min(vloss,10)):.2f}")
                if vloss < best_val:
                    best_val=vloss
                    ema.apply(model)
                    torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'loss': avg, 'val_loss': vloss}, os.path.join(cfg.checkpoint_dir, "model_resumed_best.pt"))
                    ema.restore(model)
                # sample
                model.eval()
                for prompt in ["Once upon a time, Lily", "Q: What is 12 + 8? A:"]:
                    try:
                        ids = tok.encode(prompt)
                        t = torch.tensor([ids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            out = model.generate(t, max_new_tokens=50, temperature=0.8)
                        print(f"{prompt} -> {tok.decode(out[0].tolist())[-100:]}")
                    except Exception as e:
                        print(f"sample fail {e}")
                model.train()
                # checkpoint
                ema.apply(model)
                torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'loss': avg}, os.path.join(cfg.checkpoint_dir, f"model_resumed_step{step}.pt"))
                ema.restore(model)
                print(f"Saved resumed step {step}")

    pbar.close()
    ema.apply(model)
    torch.save({'model': model.state_dict(), 'config': cfg, 'step': step, 'loss': total_loss/max(1,step)}, os.path.join(cfg.checkpoint_dir, "model_resumed_final.pt"))
    print("Done")

if __name__ == "__main__":
    train_resume()
