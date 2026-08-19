"""
Quantize TinyLLM to int8 for <2MB deployment
Uses dynamic quantization for linear layers
"""
import torch, glob, os
from .config import TinyConfig
from .model import TinyLLM

def quantize_model(ckpt_path=None, out_path="checkpoints/model_quantized_int8.pt"):
    if ckpt_path is None:
        candidates = glob.glob("checkpoints/model_*.pt")
        ckpt_path = max(candidates, key=lambda x: os.path.getmtime(x)) if candidates else None
        if not ckpt_path:
            print("No checkpoint")
            return
    print(f"Quantizing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', TinyConfig())
    model = TinyLLM(cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Dynamic quantization
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save size comparison
    orig_size = os.path.getsize(ckpt_path)/1024/1024
    # Save quantized state dict (int8)
    torch.save({'model': quantized.state_dict(), 'config': cfg, 'quantized': True, 'orig': ckpt_path}, out_path)
    q_size = os.path.getsize(out_path)/1024/1024
    print(f"Original: {orig_size:.2f} MB -> Quantized: {q_size:.2f} MB ({q_size/orig_size*100:.1f}%)")
    print(f"Saved to {out_path}")

    # Test inference
    dummy = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        logits, _ = quantized(dummy)
        print(f"Quantized inference ok: logits shape {logits.shape}")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="checkpoints/model_quantized_int8.pt")
    args=ap.parse_args()
    quantize_model(args.ckpt, args.out)
