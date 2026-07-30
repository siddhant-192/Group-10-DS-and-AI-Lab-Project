#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COLAB_BIN="$PROJECT_ROOT/.venv-colab-cli/bin/colab"
RUN_DIR="${1:-}"
INTERVAL="${SFT_MONITOR_INTERVAL:-15}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find "$PROJECT_ROOT/artifacts/qlora-training/runs" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -n 1)"
fi
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || {
  printf 'No QLoRA run directory found.\n' >&2
  exit 1
}
[[ -x "$COLAB_BIN" ]] || {
  printf 'Missing Colab CLI at %s\n' "$COLAB_BIN" >&2
  exit 1
}

SESSION_NAME="$(tr -d '\r\n' <"$RUN_DIR/session-name.txt")"
AUTH_MODE="$(tr -d '\r\n' <"$RUN_DIR/auth-mode.txt")"
STATUS_FILE="$RUN_DIR/monitor-status.json"

printf 'Monitoring QLoRA session %s every %ss\n' "$SESSION_NAME" "$INTERVAL"
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Press Ctrl-C to stop monitoring; the launcher still owns automatic shutdown.\n\n'

while true; do
  if "$COLAB_BIN" --auth="$AUTH_MODE" download -s "$SESSION_NAME" \
    "/content/text2sql_sft/status.json" "$STATUS_FILE" >/dev/null 2>&1; then
    "$PROJECT_ROOT/.venv-model-eval/bin/python" "$SCRIPT_DIR/render_sft_status.py" "$STATUS_FILE"
    PHASE="$("$PROJECT_ROOT/.venv-model-eval/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$STATUS_FILE")"
    if [[ "$PHASE" == "complete" || "$PHASE" == "failed" ]]; then
      break
    fi
  else
    printf '%s | status unavailable; checking session\n' "$(date '+%H:%M:%S')"
    if ! "$COLAB_BIN" --auth="$AUTH_MODE" status -s "$SESSION_NAME" >/dev/null 2>&1; then
      printf 'The Colab session is no longer active. Check %s/orchestrator.log\n' "$RUN_DIR" >&2
      break
    fi
  fi
  sleep "$INTERVAL"
done

printf '\nLatest launcher log lines:\n'
tail -n 30 "$RUN_DIR/orchestrator.log" 2>/dev/null || true
