"""
BPE Tokenizer for 1M model - vocab 4096
Uses huggingface tokenizers library
"""
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_tokenizer(corpus_path: str, save_path: str, vocab_size=4096):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # init tokenizer with BPE + ByteLevel pretokenizer (like GPT-2)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]","[PAD]","[BOS]","[EOS]"],
        show_progress=True,
        min_frequency=2,
    )

    print(f"Training BPE tokenizer vocab={vocab_size} on {corpus_path}")
    tokenizer.train([corpus_path], trainer)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

def load_tokenizer(path: str):
    tok = Tokenizer.from_file(path)
    return tok

class TikTokenizerWrapper:
    """Wrapper to make HF tokenizer torch-friendly"""
    def __init__(self, hf_tokenizer):
        self.tok = hf_tokenizer
        self.pad_id = self.tok.token_to_id("[PAD]") or 1
        self.bos_id = self.tok.token_to_id("[BOS]") or 2
        self.eos_id = self.tok.token_to_id("[EOS]") or 3
        self.unk_id = self.tok.token_to_id("[UNK]") or 0

    def encode(self, text: str):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    def vocab_size(self):
        return self.tok.get_vocab_size()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="checkpoints/tokenizer.json")
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()
    train_tokenizer(args.corpus, args.out, args.vocab)
