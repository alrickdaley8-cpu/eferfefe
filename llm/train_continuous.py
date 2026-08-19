"""
Continuous training without stopping — trains 1M model forever
- Loops forever over data, generating new synthetic data periodically
- Saves checkpoints every 200 steps
- Auto-resumes from latest checkpoint
- Keeps training even if interrupted (restarts)
- For 1M/20M model to become as powerful as possible over time
"""
import os, time, glob, json, math, random, torch, sys
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm import tqdm

from .config import TinyConfig
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper
from .dataset_fast import FastTokenDataset
from .build_corpus import generate_corpus

def get_lr(step, cfg, total_steps=10000):
    # Cosine with restarts every total_steps
    # Warmup 200, then cosine decay to min, then restart
    cycle = step // total_steps
    step_in_cycle = step % total_steps
    if step_in_cycle < cfg.warmup_steps:
        return cfg.lr_max * (step_in_cycle+1) / cfg.warmup_steps
    progress = (step_in_cycle - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))

def train_continuous():
    print("=== CONTINUOUS TRAINING WITHOUT STOPPING ===", flush=True)
    print("This will train forever, generating new data and improving model", flush=True)

    cfg = TinyConfig()
    cfg.batch_size = 32
    cfg.max_steps = 10000  # per cycle, but we loop forever
    cfg.warmup_steps = 200
    cfg.lr_max = 5e-4
    cfg.lr_min = 5e-5

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load tokenizer
    if not os.path.exists(cfg.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {cfg.tokenizer_path}")

    hf = Tokenizer.from_file(cfg.tokenizer_path)
    tok_wrap = TikTokenizerWrapper(hf)

    # Ensure tokens exist
    tokens_pt = "data/tokens.pt"
    if not os.path.exists(tokens_pt):
        from .preprocess import preprocess
        preprocess(cfg.data_path, cfg.tokenizer_path, tokens_pt, max_tokens=cfg.total_tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Find latest checkpoint to resume
    global_step = 0
    model = None
    candidates = glob.glob(os.path.join(cfg.checkpoint_dir, "model_*.pt"))
    if candidates:
        # Prefer continuous checkpoints, then any
        cont_cands = glob.glob(os.path.join(cfg.checkpoint_dir, "model_continuous_*.pt"))
        latest = max(cont_cands or candidates, key=lambda x: os.path.getmtime(x))
        print(f"Resuming from {latest}", flush=True)
        try:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            # Handle both v1 and v2 configs
            loaded_cfg = ckpt.get('config', cfg)
            # Use current cfg for training but load weights
            model = TinyLLM(cfg).to(device)
            try:
                model.load_state_dict(ckpt['model'])
            except Exception as e:
                print(f"Load failed {e}, trying compatible load", flush=True)
                # Try to load v1 checkpoint into v1 model
                model = TinyLLM(loaded_cfg).to(device)
                model.load_state_dict(ckpt['model'])
                # If loaded_cfg is v1, keep using v1 cfg for now
                cfg = loaded_cfg if hasattr(loaded_cfg, 'd_model') else cfg
            global_step = ckpt.get('step', 0)
            print(f"Resumed at global step {global_step}", flush=True)
        except Exception as e:
            print(f"Failed to resume {e}, starting fresh", flush=True)
            model = TinyLLM(cfg).to(device)
            global_step = 0
    else:
        print("No checkpoint, starting fresh", flush=True)
        model = TinyLLM(cfg).to(device)

    print(f"Model params: {model.count_params():,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, betas=(0.9,0.95), weight_decay=0.1)

    # Training loop forever
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*20} CYCLE {cycle} START (global step {global_step}) {'='*20}", flush=True)

        # Occasionally regenerate corpus for freshness (every 3 cycles)
        if cycle % 3 == 0:
            print("Regenerating fresh corpus for continuous learning...", flush=True)
            new_corpus = f"data/corpus_continuous_{cycle}.txt"
            try:
                generate_corpus(new_corpus, target_chars=20_000_000)  # 20M chars ~5M tokens fresh
                # Preprocess to tokens and mix with existing
                from .preprocess import preprocess
                fresh_tokens = f"data/tokens_fresh_{cycle}.pt"
                preprocess(new_corpus, cfg.tokenizer_path, fresh_tokens, max_tokens=5_000_000)
                print(f"Fresh tokens generated: {fresh_tokens}", flush=True)
                # Use fresh tokens for this cycle
                dataset_path = fresh_tokens
            except Exception as e:
                print(f"Fresh corpus failed {e}, using main tokens", flush=True)
                dataset_path = tokens_pt
        else:
            dataset_path = tokens_pt

        # Load dataset
        try:
            full_ds = FastTokenDataset(dataset_path, seq_len=cfg.max_seq_len)
        except Exception as e:
            print(f"Dataset load failed {e}, using main", flush=True)
            full_ds = FastTokenDataset(tokens_pt, seq_len=cfg.max_seq_len)

        # Train/val split
        n_val = int(len(full_ds)*0.02)
        n_train = len(full_ds) - n_val
        from torch.utils.data import Subset
        train_ds = Subset(full_ds, range(n_train))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        data_iter = iter(train_loader)
        model.train()
        cycle_steps = 0
        cycle_loss = 0

        # Train for cfg.max_steps or until dataset exhausted
        pbar = tqdm(total=cfg.max_steps, desc=f"Cycle {cycle} Continuous", initial=0)
        while cycle_steps < cfg.max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)

            lr = get_lr(global_step, cfg, total_steps=cfg.max_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            cycle_loss += loss.item()
            global_step += 1
            cycle_steps += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "avg": f"{cycle_loss/cycle_steps:.3f}",
                "lr": f"{lr:.2e}",
                "g_step": global_step,
                "tok": f"{global_step*cfg.tokens_per_batch()/1e6:.1f}M"
            })
            pbar.update(1)

            # Save every 200 steps
            if global_step % 200 == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_continuous_step{global_step}.pt")
                torch.save({
                    'model': model.state_dict(),
                    'config': cfg,
                    'step': global_step,
                    'loss': cycle_loss/cycle_steps,
                    'cycle': cycle
                }, ckpt_path)
                print(f"\n[Continuous] Saved {ckpt_path} | global_step {global_step} | loss {cycle_loss/cycle_steps:.4f} | tokens {global_step*cfg.tokens_per_batch()/1e6:.1f}M", flush=True)

                # Also save as latest for server auto-reload
                latest_path = os.path.join(cfg.checkpoint_dir, f"model_chat_final.pt")
                # Don't overwrite if chat fine-tune is more recent? We save as continuous_final to avoid conflict
                cont_final = os.path.join(cfg.checkpoint_dir, f"model_continuous_final.pt")
                torch.save({
                    'model': model.state_dict(),
                    'config': cfg,
                    'step': global_step,
                    'loss': cycle_loss/cycle_steps
                }, cont_final)

                # Log
                log_path = os.path.join(cfg.checkpoint_dir, "continuous_log.jsonl")
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        "global_step": global_step,
                        "cycle": cycle,
                        "cycle_step": cycle_steps,
                        "loss": loss.item(),
                        "avg_loss": cycle_loss/cycle_steps,
                        "lr": lr,
                        "tokens_m": global_step*cfg.tokens_per_batch()/1e6,
                        "time": time.time()
                    })+"\n")

        pbar.close()
        print(f"Cycle {cycle} done, avg loss {cycle_loss/cycle_steps:.4f}, global {global_step}", flush=True)
        # Brief pause to let server reload and avoid overheating
        time.sleep(2)

if __name__ == "__main__":
    # Loop forever even if crashes
    while True:
        try:
            train_continuous()
        except KeyboardInterrupt:
            print("Continuous training stopped by user", flush=True)
            break
        except Exception as e:
            import traceback
            print(f"Error in continuous training: {e}", flush=True)
            traceback.print_exc()
            print("Restarting in 5 seconds...", flush=True)
            time.sleep(5)
            continue
