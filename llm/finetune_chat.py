"""
Fine-tune pretrained TinyLLM on chat instruction data
Takes base checkpoint and trains for a few hundred steps on chat format
"""
import os, json, torch, glob, math, time
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from .config import TinyConfig
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper
from tqdm import tqdm

class ChatDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len
        self.examples = []
        print(f"Loading chat data {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    msgs = ex.get('messages', [])
                    # format as chat prompt
                    prompt = ""
                    for m in msgs:
                        role = m.get('role','user')
                        content = m.get('content','')
                        if role=='system':
                            prompt += f"System: {content}\n"
                        elif role=='user':
                            prompt += f"User: {content}\n"
                        elif role=='assistant':
                            prompt += f"Assistant: {content}\n"
                    # For training, we want to predict assistant part only? For simplicity, train on full sequence causal LM
                    ids = self.tok.encode(prompt)
                    if len(ids) < 20: continue
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                    self.examples.append(ids)
                except Exception as e:
                    continue
        print(f"Loaded {len(self.examples)} chat examples")

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ids = self.examples[idx]
        # pad/truncate to max_len
        if len(ids) < self.max_len:
            pad_id = self.tok.pad_id
            ids = ids + [pad_id]*(self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x,y

def finetune(base_ckpt="checkpoints/model_step600.pt", chat_data="data/chat_finetune.jsonl", out_dir="checkpoints", steps=300, lr=1e-4):
    cfg = TinyConfig()
    cfg.max_steps = steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(cfg.tokenizer_path)
    tok = TikTokenizerWrapper(hf)

    # ensure chat data exists
    if not os.path.exists(chat_data):
        print(f"Chat data not found, generating...")
        from .chat_data import generate_chat_dataset
        generate_chat_dataset(10000, chat_data)

    dataset = ChatDataset(chat_data, tok, max_len=cfg.max_seq_len)
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # load base model
    if base_ckpt and os.path.exists(base_ckpt):
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get('config', cfg)
        model = TinyLLM(cfg).to(device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded base {base_ckpt}")
    else:
        # latest
        candidates = glob.glob("checkpoints/model_*.pt")
        if candidates:
            latest = max(candidates, key=lambda x: os.path.getmtime(x))
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            cfg = ckpt.get('config', cfg)
            model = TinyLLM(cfg).to(device)
            model.load_state_dict(ckpt['model'])
            print(f"Loaded latest {latest}")
        else:
            model = TinyLLM(cfg).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)

    it = iter(dl)
    pbar = tqdm(total=steps)
    total_loss=0
    for step in range(steps):
        try:
            x,y = next(it)
        except StopIteration:
            it = iter(dl)
            x,y = next(it)
        x=x.to(device); y=y.to(device)
        logits, loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss+=loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "avg": f"{total_loss/(step+1):.3f}"})
        pbar.update(1)
        if (step+1)%100==0:
            path = os.path.join(out_dir, f"model_chat_ft_step{step+1}.pt")
            torch.save({'model': model.state_dict(), 'config': cfg, 'step': step+1, 'loss': total_loss/(step+1)}, path)
            print(f"Saved {path}")

    pbar.close()
    final = os.path.join(out_dir, "model_chat_final.pt")
    torch.save({'model': model.state_dict(), 'config': cfg, 'step': steps, 'loss': total_loss/steps}, final)
    print(f"Chat fine-tune done -> {final}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--data", default="data/chat_finetune.jsonl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    finetune(base_ckpt=args.base, chat_data=args.data, steps=args.steps, lr=args.lr)
