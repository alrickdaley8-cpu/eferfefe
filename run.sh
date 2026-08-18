#!/bin/bash
set -e

VENV=.venv
if [ ! -d "$VENV" ]; then
  echo "Creating venv..."
  python3 -m venv $VENV
fi

source $VENV/bin/activate
pip install -q --upgrade pip
pip install -q torch tokenizers tqdm numpy

echo "== TinyLLM 1M / 20M Token Pipeline =="
echo "Config: 1.06M params, 20M tokens (8192 tok/batch, 2442 steps)"

if [ ! -f data/corpus.txt ]; then
  echo "Building corpus (85M chars ~20M tokens) ..."
  python -m llm.build_corpus --out data/corpus.txt --chars 85000000
else
  echo "Corpus exists: $(du -h data/corpus.txt | cut -f1)"
fi

if [ ! -f checkpoints/tokenizer.json ]; then
  echo "Training tokenizer vocab 4096..."
  python -m llm.tokenizer --corpus data/corpus.txt --out checkpoints/tokenizer.json --vocab 4096
else
  echo "Tokenizer exists"
fi

echo "Starting training..."
python -m llm.train

echo "Training done, testing generate..."
python -m llm.generate --prompt "Once upon a time" --tokens 100

echo "Done! Model at checkpoints/model_final.pt"
