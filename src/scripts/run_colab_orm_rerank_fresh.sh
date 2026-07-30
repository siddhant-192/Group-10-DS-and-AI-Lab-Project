#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB_BIN="$PROJECT_ROOT/.venv-colab-cli/bin/colab"
REMOTE_REQUIREMENTS="$PROJECT_ROOT/scripts/colab-eval-requirements.txt"
PREPARE_SCRIPT="$PROJECT_ROOT/scripts/colab_prepare_orm_rerank.py"
LAUNCH_SCRIPT="$PROJECT_ROOT/scripts/colab_launch_orm_rerank.py"
TRAINING_RUN=""
CANDIDATE_GROUPS=""
AUTH_MODE="adc"
ASSUME_YES=0
SESSION_ACTIVE=0
STOPPED=0
SESSION_NAME="orm-rerank-$(date +%Y%m%d-%H%M%S)-$$"

usage() {
  cat <<'EOF'
Rerank SQL candidates with a completed ORM adapter on a fresh Colab L4.

Usage:
  bash scripts/run_colab_orm_rerank_fresh.sh \
    --training-run DIR --candidate-groups FILE [--yes]

The adapter is uploaded from the validated local training run. Results are
downloaded and the L4 is stopped automatically, including on failure.
EOF
}

die() { printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[0;34mINFO:\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33mWARNING:\033[0m %s\n' "$*" >&2; }

while (($#)); do
  case "$1" in
    --training-run) TRAINING_RUN="${2:-}"; shift 2 ;;
    --candidate-groups) CANDIDATE_GROUPS="${2:-}"; shift 2 ;;
    --session) SESSION_NAME="${2:-}"; shift 2 ;;
    --auth) AUTH_MODE="${2:-}"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -x "$COLAB_BIN" ]] || die "Colab CLI not found: $COLAB_BIN"
[[ -n "$TRAINING_RUN" && -d "$TRAINING_RUN" ]] || die "Invalid --training-run"
[[ -n "$CANDIDATE_GROUPS" && -f "$CANDIDATE_GROUPS" ]] || die "Invalid --candidate-groups"
[[ "$AUTH_MODE" == "adc" || "$AUTH_MODE" == "oauth2" ]] || die "--auth must be adc or oauth2"
[[ "$SESSION_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "Invalid session name"
for required in "$REMOTE_REQUIREMENTS" "$PREPARE_SCRIPT" "$LAUNCH_SCRIPT" \
  "$PROJECT_ROOT/scripts/evaluate_text2sql_models.py" \
  "$PROJECT_ROOT/scripts/evaluate_orm_candidate_groups.py"; do
  [[ -f "$required" ]] || die "Missing required file: $required"
done

TRAINING_RUN="$(cd "$TRAINING_RUN" && pwd)"
CANDIDATE_GROUPS="$(cd "$(dirname "$CANDIDATE_GROUPS")" && pwd)/$(basename "$CANDIDATE_GROUPS")"
ADAPTER_DIR="$TRAINING_RUN/downloaded/output/final_adapter"
[[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || die "Validated adapter is missing"
[[ -f "$TRAINING_RUN/artifact-validation.json" ]] || die "Training artifact validation is missing"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$PROJECT_ROOT/artifacts/orm-reranking/runs/$RUN_STAMP-fresh"
BUNDLE="$RUN_DIR/orm_rerank_bundle.tar.gz"
mkdir -p "$RUN_DIR"
printf '%s\n' "$SESSION_NAME" > "$RUN_DIR/session-name.txt"
printf '%s\n' "$TRAINING_RUN" > "$RUN_DIR/training-run.txt"
printf '%s\n' "$CANDIDATE_GROUPS" > "$RUN_DIR/candidate-groups.txt"

colab_cmd() { "$COLAB_BIN" --auth="$AUTH_MODE" "$@"; }
stop_session() {
  if (( SESSION_ACTIVE )) && (( ! STOPPED )); then
    info "Stopping L4 session '$SESSION_NAME'."
    if colab_cmd stop -s "$SESSION_NAME"; then
      STOPPED=1
      SESSION_ACTIVE=0
    else
      warn "Automatic stop failed: $COLAB_BIN --auth=$AUTH_MODE stop -s $SESSION_NAME"
    fi
  fi
}
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  stop_session
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

if (( ! ASSUME_YES )); then
  [[ -t 0 ]] || die "Confirmation requires a terminal; rerun with --yes."
  read -r -p "Create an L4 session for ORM reranking? Type 'yes': " reply
  [[ "$reply" == "yes" ]] || die "Cancelled."
fi

info "Building the adapter/config bundle."
tar -czf "$BUNDLE" \
  -C "$TRAINING_RUN/downloaded" output/final_adapter \
  -C "$PROJECT_ROOT" configs/text2sql_eval_models.json models/text2sql-eval/download_manifest.json

info "Creating fresh Colab L4 session '$SESSION_NAME'."
SESSION_ACTIVE=1
colab_cmd new -s "$SESSION_NAME" --gpu L4 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
colab_cmd status -s "$SESSION_NAME" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"

install_ok=0
for attempt in 1 2 3; do
  if colab_cmd install -s "$SESSION_NAME" -r "$REMOTE_REQUIREMENTS" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; then
    install_ok=1
    break
  fi
  warn "Dependency installation attempt $attempt failed."
  sleep 10
done
(( install_ok )) || die "Could not install reranking dependencies"

upload_with_retry() {
  local source="$1"
  local destination="$2"
  local attempt
  for attempt in 1 2 3; do
    if colab_cmd upload -s "$SESSION_NAME" "$source" "$destination" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; then
      return 0
    fi
    warn "Upload attempt $attempt failed for $(basename "$source")."
    sleep 5
  done
  return 1
}

info "Uploading adapter, candidates, and reranking code."
upload_with_retry "$BUNDLE" /content/orm_rerank_bundle.tar.gz || die "Bundle upload failed"
upload_with_retry "$PROJECT_ROOT/scripts/evaluate_text2sql_models.py" /content/evaluate_text2sql_models.py || die "Core evaluator upload failed"
upload_with_retry "$PROJECT_ROOT/scripts/evaluate_orm_candidate_groups.py" /content/evaluate_orm_candidate_groups.py || die "ORM evaluator upload failed"
upload_with_retry "$CANDIDATE_GROUPS" /content/orm_candidate_groups.jsonl || die "Candidate upload failed"

info "Verifying the fresh reranking workspace."
colab_cmd exec -s "$SESSION_NAME" -f "$PREPARE_SCRIPT" --timeout 900 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
info "Running ORM reranking."
colab_cmd exec -s "$SESSION_NAME" -f "$LAUNCH_SCRIPT" --timeout 7200 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
colab_cmd download -s "$SESSION_NAME" /content/orm_rerank_results.tar.gz "$RUN_DIR/results.tar.gz" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
mkdir -p "$RUN_DIR/downloaded"
tar -xzf "$RUN_DIR/results.tar.gz" -C "$RUN_DIR/downloaded"
colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb" || true
stop_session
trap - EXIT INT TERM

printf '\n\033[1;32mFresh ORM reranking completed.\033[0m\n'
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Results: %s\n' "$RUN_DIR/downloaded/results"
