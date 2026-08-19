#!/usr/bin/env bash
# One command from a bare clone to a fully working tiny-lm: environment, corpus, tokenizer,
# tokenised data, knowledge base, instruction + reasoning sets, then the background trainer.
#
#   ai/bootstrap.sh              # everything, then start training in the background
#   ai/bootstrap.sh --data-only  # build the artefacts but do not start training
#   ai/bootstrap.sh --quick      # small corpus (fast smoke test of the whole pipeline)
#
# Every step is skipped when its output already exists, so this is safe to re-run — which matters
# because ai/data and ai/checkpoints are git-ignored and vanish when a sandbox is recycled.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/.venv}"
PY="$VENV/bin/python"
DATA="$ROOT/ai/data"
QA="$DATA/qa"

QUICK=0; DATA_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --data-only) DATA_ONLY=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

CORPUS_MB=450; PACKAGES=12000; TOKENS=100000000
if [[ $QUICK == 1 ]]; then CORPUS_MB=25; PACKAGES=800; TOKENS=4000000; fi

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "python environment ($VENV)"
if [[ ! -x "$PY" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
fi
"$VENV/bin/pip" install -q -r "$ROOT/ai/requirements.txt"
"$PY" -c "import torch, tokenizers, numpy, requests; print('torch', torch.__version__)"

cd "$ROOT"
export PYTHONPATH="$ROOT"

say "corpus"
if [[ -d "$DATA/corpus" ]] && compgen -G "$DATA/corpus/shard_*.txt" > /dev/null; then
  echo "already built ($(du -sh "$DATA/corpus" | cut -f1))"
else
  "$PY" ai/build_corpus.py --target-mb "$CORPUS_MB" --workers 24
fi

say "tokenizer"
[[ -f "$DATA/tokenizer.json" ]] && echo "already built" || "$PY" ai/train_tokenizer.py

say "tokenised pretraining data"
if [[ -f "$DATA/train.bin" ]]; then
  echo "already built ($(( $(stat -c%s "$DATA/train.bin") / 2 )) tokens)"
else
  "$PY" ai/prepare_data.py --max-tokens "$TOKENS"
fi

say "knowledge base + instruction data"
if [[ -f "$QA/sft.jsonl" ]]; then
  echo "instruction set already built"
elif [[ -f "$QA/knowledge.jsonl" ]]; then
  echo "knowledge base is committed — regenerating the instruction set offline"
  "$PY" ai/build_qa.py --from-knowledge
else
  "$PY" ai/build_qa.py --packages "$PACKAGES" --workers 32
fi
[[ -f "$QA/reasoning.jsonl" ]] && echo "reasoning set already built" \
  || "$PY" ai/build_reasoning.py

say "packing instruction tokens"
if [[ -f "$QA/sft_tokens.bin" ]]; then
  echo "already packed"
else
  "$PY" - <<EOF
from tokenizers import Tokenizer
from ai.finetune import pack
import os
qa = "$QA"
pack(Tokenizer.from_file("$DATA/tokenizer.json"),
     [os.path.join(qa, "sft.jsonl"), os.path.join(qa, "reasoning.jsonl")],
     os.path.join(qa, "sft_tokens.bin"), os.path.join(qa, "sft_mask.bin"))
EOF
fi

say "restoring a published checkpoint, if one is committed"
"$ROOT/ai/publish_model.sh" restore || true

if [[ $DATA_ONLY == 1 ]]; then
  say "done (data only)"
  exit 0
fi

say "starting the training daemon"
"$ROOT/ai/daemon.sh" start
sleep 3
"$ROOT/ai/daemon.sh" status || true

cat <<'EOF'

ready:
  ai/daemon.sh status          progress of the background run
  python -m ai.serve --port 8000   site + chat  (http://localhost:8000)
  python -m ai.chat --think "does fastapi depend on pydantic?"
  python -m ai.tests           full test suite
  python -m ai.eval --n 60     answer-accuracy benchmark
EOF
