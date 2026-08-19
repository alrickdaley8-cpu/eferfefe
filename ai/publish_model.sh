#!/usr/bin/env bash
# Checkpoints live in git-ignored ai/checkpoints/ and disappear when a machine is recycled.
# This publishes the current best model into git-tracked ai/release/ (a ~20 MB weights file plus
# a small metadata json) so a fresh clone can chat immediately, and restores it on bootstrap.
#
#   ai/publish_model.sh publish [file]   copy ai/checkpoints/<file> -> ai/release/model.pt
#   ai/publish_model.sh restore          copy it back if no live checkpoint exists
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/ai/checkpoints"
REL="$ROOT/ai/release"
PY="${PY:-$HOME/.venv/bin/python}"

case "${1:-publish}" in
  publish)
    src="$CKPT/${2:-}"
    if [[ -z "${2:-}" ]]; then
      for cand in sft.pt sft_demo.pt best.pt model.pt; do
        [[ -f "$CKPT/$cand" ]] && { src="$CKPT/$cand"; break; }
      done
    fi
    [[ -f "$src" ]] || { echo "no checkpoint to publish"; exit 1; }
    mkdir -p "$REL"
    cp "$src" "$REL/model.pt"
    "$PY" - "$src" "$REL/meta.json" <<'PYEOF'
import json, os, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
json.dump({"source": os.path.basename(sys.argv[1]), "stage": ck.get("stage", "pretrain"),
           "step": ck.get("step", 0), "tokens_seen": ck.get("step", 0) * 8192,
           "published_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       __import__("time").gmtime())},
          open(sys.argv[2], "w"), indent=2)
PYEOF
    echo "published $(basename "$src") -> ai/release/model.pt ($(du -h "$REL/model.pt" | cut -f1))"
    ;;
  restore)
    [[ -f "$REL/model.pt" ]] || { echo "nothing published yet"; exit 0; }
    mkdir -p "$CKPT"
    if compgen -G "$CKPT/*.pt" > /dev/null; then echo "live checkpoints exist, not restoring"; exit 0; fi
    cp "$REL/model.pt" "$CKPT/released.pt"
    echo "restored ai/release/model.pt -> ai/checkpoints/released.pt"
    ;;
  *) echo "usage: $0 {publish|restore} [checkpoint]"; exit 2 ;;
esac
