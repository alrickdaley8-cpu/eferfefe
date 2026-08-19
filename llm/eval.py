"""
Evaluate perplexity on validation set
"""
import torch, glob, os, math
from tokenizers import Tokenizer
from .config import TinyConfig
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper
from .dataset_fast import FastTokenDataset
from torch.utils.data import DataLoader, Subset

def eval_perplexity(ckpt_path=None, tokens_path="data/tokens.pt", val_split=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is None:
        candidates = glob.glob("checkpoints/model_*.pt")
        if not candidates:
            print("No checkpoint")
            return
        ckpt_path = max(candidates, key=lambda x: os.path.getmtime(x))
    print(f"Evaluating {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', TinyConfig())
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    hf = Tokenizer.from_file(cfg.tokenizer_path)
    tok = TikTokenizerWrapper(hf)

    dataset = FastTokenDataset(tokens_path, seq_len=cfg.max_seq_len)
    n_val = int(len(dataset)*val_split)
    n_train = len(dataset)-n_val
    val_ds = Subset(dataset, range(n_train, n_train+n_val))
    dl = DataLoader(val_ds, batch_size=32)

    total_loss=0
    total_tokens=0
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device); y=y.to(device)
            logits, loss = model(x,y)
            total_loss += loss.item()*x.numel()
            total_tokens += x.numel()
    avg_loss = total_loss/total_tokens
    ppl = math.exp(avg_loss)
    print(f"Val loss: {avg_loss:.4f} | Perplexity: {ppl:.2f} on {n_val} seqs ({n_val*cfg.max_seq_len} tokens)")
    return avg_loss, ppl

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokens", default="data/tokens.pt")
    args=ap.parse_args()
    eval_perplexity(args.ckpt, args.tokens)
