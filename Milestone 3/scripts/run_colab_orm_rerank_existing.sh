#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB_BIN="$PROJECT_ROOT/.venv-colab-cli/bin/colab"
TRAINING_RUN=""
CANDIDATE_GROUPS=""
AUTH_MODE="adc"

usage() {
  cat <<'EOF'
Rerank SQL candidates with a completed ORM adapter in its retained Colab session.

Usage:
  bash scripts/run_colab_orm_rerank_existing.sh --training-run DIR --candidate-groups FILE

The training launcher must have used --keep-session. Results are downloaded and
the L4 is stopped automatically, including when reranking fails.
EOF
}

die() { printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[0;34mINFO:\033[0m %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --training-run) TRAINING_RUN="${2:-}"; shift 2 ;;
    --candidate-groups) CANDIDATE_GROUPS="${2:-}"; shift 2 ;;
    --auth) AUTH_MODE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -x "$COLAB_BIN" ]] || die "Colab CLI not found: $COLAB_BIN"
[[ -n "$TRAINING_RUN" && -d "$TRAINING_RUN" ]] || die "Invalid --training-run"
[[ -n "$CANDIDATE_GROUPS" && -f "$CANDIDATE_GROUPS" ]] || die "Invalid --candidate-groups"
[[ "$AUTH_MODE" == "adc" || "$AUTH_MODE" == "oauth2" ]] || die "--auth must be adc or oauth2"

TRAINING_RUN="$(cd "$TRAINING_RUN" && pwd)"
CANDIDATE_GROUPS="$(cd "$(dirname "$CANDIDATE_GROUPS")" && pwd)/$(basename "$CANDIDATE_GROUPS")"
SESSION_FILE="$TRAINING_RUN/session-name.txt"
ADAPTER_FILE="$TRAINING_RUN/downloaded/output/final_adapter/adapter_model.safetensors"
[[ -f "$SESSION_FILE" ]] || die "Missing session name: $SESSION_FILE"
[[ -f "$ADAPTER_FILE" ]] || die "Training results are not complete: $ADAPTER_FILE"
SESSION_NAME="$(tr -d '[:space:]' < "$SESSION_FILE")"
[[ -n "$SESSION_NAME" ]] || die "Session name is empty"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$PROJECT_ROOT/artifacts/orm-reranking/runs/$RUN_STAMP"
mkdir -p "$RUN_DIR"
printf '%s\n' "$SESSION_NAME" > "$RUN_DIR/session-name.txt"
printf '%s\n' "$TRAINING_RUN" > "$RUN_DIR/training-run.txt"

colab_cmd() { "$COLAB_BIN" --auth="$AUTH_MODE" "$@"; }
stopped=0
cleanup() {
  trap - EXIT INT TERM
  if (( stopped == 0 )); then
    info "Stopping retained L4 session '$SESSION_NAME'."
    colab_cmd stop -s "$SESSION_NAME" || true
  fi
}
trap cleanup EXIT INT TERM

info "Verifying retained training session '$SESSION_NAME'."
colab_cmd status -s "$SESSION_NAME"
info "Uploading gold-blind candidates and reranking code."
colab_cmd upload -s "$SESSION_NAME" "$PROJECT_ROOT/scripts/evaluate_text2sql_models.py" \
  /content/evaluate_text2sql_models.py
colab_cmd upload -s "$SESSION_NAME" "$PROJECT_ROOT/scripts/evaluate_orm_candidate_groups.py" \
  /content/evaluate_orm_candidate_groups.py
colab_cmd upload -s "$SESSION_NAME" "$CANDIDATE_GROUPS" /content/orm_candidate_groups.jsonl

info "Running ORM reranking."
colab_cmd exec -s "$SESSION_NAME" -f "$PROJECT_ROOT/scripts/colab_launch_orm_rerank.py" \
  --timeout 7200 2>&1 | tee "$RUN_DIR/orchestrator.log"
colab_cmd download -s "$SESSION_NAME" /content/orm_rerank_results.tar.gz \
  "$RUN_DIR/results.tar.gz"
mkdir -p "$RUN_DIR/downloaded"
tar -xzf "$RUN_DIR/results.tar.gz" -C "$RUN_DIR/downloaded"
colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb" || true
colab_cmd stop -s "$SESSION_NAME"
stopped=1
trap - EXIT INT TERM

printf '\n\033[1;32mORM reranking completed.\033[0m\n'
printf 'Results: %s\n' "$RUN_DIR/downloaded/results"
