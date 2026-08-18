"""
Inference / generation script
"""
import os
import torch
from tokenizers import Tokenizer
from .config import TinyConfig
from .model import TinyLLM
from .tokenizer import TikTokenizerWrapper

def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', TinyConfig())
    # if cfg is dataclass from file, keep
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    wrapper = TikTokenizerWrapper(tok)
    return model, wrapper, cfg

def generate_prompt(model, tokenizer, prompt, device, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.92):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    text = tokenizer.decode(out[0].tolist())
    return text

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model_final.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time, Lily")
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        # try find latest
        import glob
        files = glob.glob("checkpoints/model_*.pt")
        if files:
            args.ckpt = sorted(files)[-1]
            print(f"Using latest checkpoint {args.ckpt}")
        else:
            raise FileNotFoundError(f"No checkpoint at {args.ckpt}")

    model, tok, cfg = load_model(args.ckpt, device)
    print(f"Model params: {model.count_params():,}")
    out = generate_prompt(model, tok, args.prompt, device, max_new_tokens=args.tokens, temperature=args.temp)
    print("="*60)
    print(f"PROMPT: {args.prompt}")
    print("-"*60)
    print(out)
    print("="*60)
