#!/usr/bin/env bash
# Supervisor that keeps the whole training pipeline running in the background.
#
#   ai/daemon.sh start     # detach and run: pretrain (100M tokens) -> instruction tuning (15M)
#   ai/daemon.sh status    # what is running, how far along
#   ai/daemon.sh stop      # graceful stop (the trainer checkpoints before exiting)
#   ai/daemon.sh restart
#   ai/daemon.sh logs      # follow the log
#
# The daemon is detached with setsid, restarts a crashed stage with exponential backoff, and
# every stage resumes from its last checkpoint, so stopping and starting is always safe.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/ai/checkpoints"
PY="${PY:-$HOME/.venv/bin/python}"
PIDFILE="$CKPT/daemon.pid"
LOG="$CKPT/daemon.log"

PRETRAIN_TOKENS="${PRETRAIN_TOKENS:-100000000}"
SFT_TOKENS="${SFT_TOKENS:-15000000}"
MAX_BACKOFF=300
SERVE_PORT="${SERVE_PORT:-8000}"

mkdir -p "$CKPT"

is_running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

stage_done() { [[ -f "$CKPT/$1.done" ]]; }   # markers written by the trainers themselves

port_busy() { "$PY" - "$1" <<'EOF' 2>/dev/null
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
EOF
}

serve_watchdog() {
  # keep the chat UI alive too, but never fight an already-running server
  while true; do
    if ! port_busy "$SERVE_PORT"; then
      echo "[daemon] starting chat server on :$SERVE_PORT"
      nice -n 10 "$PY" -m ai.serve --port "$SERVE_PORT" --threads 1 >>"$CKPT/serve.log" 2>&1 &
    fi
    sleep 60
  done
}

supervise() {
  echo "[daemon] started pid $$ at $(date -Is)"
  [[ "${SERVE:-1}" == "1" ]] && serve_watchdog &
  local backoff=5
  while true; do
    if ! stage_done pretrain; then
      echo "[daemon] --- stage: pretrain ($PRETRAIN_TOKENS tokens) ---"
      nice -n 5 "$PY" -m ai.train --total-tokens "$PRETRAIN_TOKENS" \
        --log-interval 25 --eval-interval 250 --ckpt-interval 250
      rc=$?
    elif ! stage_done sft; then
      echo "[daemon] --- stage: instruction tuning ($SFT_TOKENS tokens) ---"
      nice -n 5 "$PY" -m ai.finetune --tokens "$SFT_TOKENS" \
        --base "$CKPT/model.pt" --out "$CKPT/sft.pt" --ckpt-interval 200
      rc=$?
    else
      echo "[daemon] pipeline complete at $(date -Is) — idling (stop me with ai/daemon.sh stop)"
      while true; do sleep 3600; done
    fi

    if [[ $rc -eq 0 ]]; then
      backoff=5
      continue                      # stage finished or checkpointed cleanly; loop picks the next one
    fi
    echo "[daemon] stage exited with code $rc — restarting from checkpoint in ${backoff}s"
    sleep "$backoff"
    backoff=$(( backoff * 2 )); (( backoff > MAX_BACKOFF )) && backoff=$MAX_BACKOFF
  done
}

case "${1:-start}" in
  start)
    if is_running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    cd "$ROOT"
    PYTHONPATH="$ROOT" setsid nohup "$0" __supervise >>"$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 1
    echo "training daemon started (pid $(cat "$PIDFILE")), log: $LOG"
    ;;
  __supervise)
    cd "$ROOT"
    export PYTHONPATH="$ROOT"
    echo $$ > "$PIDFILE"
    trap 'echo "[daemon] terminating children"; pkill -TERM -P $$; sleep 5; exit 0' TERM INT
    supervise
    ;;
  stop)
    if ! is_running; then echo "not running"; rm -f "$PIDFILE"; exit 0; fi
    pid="$(cat "$PIDFILE")"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 30); do is_running || break; sleep 1; done
    is_running && kill -KILL -- "-$pid" 2>/dev/null
    rm -f "$PIDFILE"
    echo "stopped (checkpoint saved)"
    ;;
  restart) "$0" stop; sleep 2; "$0" start ;;
  status)
    if is_running; then echo "daemon: running (pid $(cat "$PIDFILE"))"; else echo "daemon: stopped"; fi
    cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" -m ai.status
    ;;
  logs) tail -f "$LOG" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}"; exit 2 ;;
esac
