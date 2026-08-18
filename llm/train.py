"""
Train 1M model on 20M tokens
"""
import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from .config import TinyConfig
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper
from tokenizers import Tokenizer
from .dataset import TextDataset
from .dataset_fast import FastTokenDataset
from .preprocess import preprocess as preprocess_fn

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr_max * (step+1) / cfg.warmup_steps
    # cosine decay
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))

def train():
    cfg = TinyConfig()
    print(f"=== TinyLLM 1M / 20M tokens ===")
    print(f"Config: d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} vocab={cfg.vocab_size}")
    print(f"Param estimate: {cfg.param_count():,} (tied={cfg.tie_weights})")
    tokens_per_batch = cfg.tokens_per_batch()
    total_tokens = cfg.max_steps * tokens_per_batch
    print(f"Tokens/batch: {tokens_per_batch:,} -> Steps {cfg.max_steps} => total {total_tokens:,} tokens (target {cfg.total_tokens:,})")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # check corpus
    if not os.path.exists(cfg.data_path):
        print(f"Corpus not found at {cfg.data_path}, generating...")
        from .build_corpus import generate_corpus
        generate_corpus(cfg.data_path, target_chars=85_000_000)

    # tokenizer
    if not os.path.exists(cfg.tokenizer_path):
        print(f"Tokenizer not found at {cfg.tokenizer_path}, training...")
        from .tokenizer import train_tokenizer
        train_tokenizer(cfg.data_path, cfg.tokenizer_path, vocab_size=cfg.vocab_size)
    else:
        print(f"Tokenizer found at {cfg.tokenizer_path}")

    hf_tok = Tokenizer.from_file(cfg.tokenizer_path)
    tok_wrap = TikTokenizerWrapper(hf_tok)
    print(f"Tokenizer vocab: {tok_wrap.vocab_size()}", flush=True)

    # pre-tokenized fast path
    tokens_pt = "data/tokens.pt"
    if not os.path.exists(tokens_pt):
        print(f"Pre-tokenized file not found at {tokens_pt}, creating 20M token slice...", flush=True)
        preprocess_fn(cfg.data_path, cfg.tokenizer_path, tokens_pt, max_tokens=cfg.total_tokens)
    else:
        print(f"Using pre-tokenized file {tokens_pt}", flush=True)

    # dataset - try fast first
    try:
        dataset = FastTokenDataset(tokens_pt, seq_len=cfg.max_seq_len)
    except Exception as e:
        print(f"Fast dataset failed {e}, falling back to TextDataset", flush=True)
        dataset = TextDataset(cfg.data_path, tok_wrap, seq_len=cfg.max_seq_len)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TinyLLM(cfg).to(device)
    actual_params = model.count_params()
    print(f"Actual model params: {actual_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, betas=cfg.betas, weight_decay=cfg.weight_decay)

    # training loop
    model.train()
    step = 0
    total_loss = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    pbar = tqdm(total=cfg.max_steps, desc="Training")
    while step < cfg.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "avg": f"{total_loss/(step+1):.3f}", "lr": f"{lr:.2e}", "tok": f"{(step+1)*tokens_per_batch/1e6:.2f}M"})
        pbar.update(1)
        step += 1

        if step % 200 == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_step{step}.pt")
            torch.save({
                'model': model.state_dict(),
                'config': cfg,
                'step': step,
                'loss': total_loss/(step+1)
            }, ckpt_path)
            print(f"\nCheckpoint saved to {ckpt_path}")

            # quick generate sample
            model.eval()
            prompt = "Once upon a time"
            prompt_ids = tok_wrap.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(prompt_tensor, max_new_tokens=80, temperature=0.8, top_k=50)
                decoded = tok_wrap.decode(out[0].tolist())
                print(f"\n--- SAMPLE (step {step}) ---\nPrompt: {prompt}\nGenerated: {decoded}\n--- END SAMPLE ---\n")
            model.train()

        if step % 500 == 0:
            print(f"Step {step}: avg loss {total_loss/(step+1):.4f} | elapsed { (time.time()-start_time)/60:.1f} min | tokens {(step*tokens_per_batch)/1e6:.2f}M")

    pbar.close()
    # final save
    final_path = os.path.join(cfg.checkpoint_dir, "model_final.pt")
    torch.save({
        'model': model.state_dict(),
        'config': cfg,
        'step': step,
        'loss': total_loss/step
    }, final_path)
    print(f"Training done! Final model saved to {final_path}")
    print(f"Total tokens trained: {step*tokens_per_batch:,}")

if __name__ == "__main__":
    train()
