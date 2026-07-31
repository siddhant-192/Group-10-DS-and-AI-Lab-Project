#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COLAB_BIN="$PROJECT_ROOT/.venv-colab-cli/bin/colab"
BUILD_PYTHON="$PROJECT_ROOT/.venv-model-eval/bin/python"
REMOTE_REQUIREMENTS="$SCRIPT_DIR/colab-sft-requirements.txt"
PREPARE_SCRIPT="$SCRIPT_DIR/colab_prepare_qlora_sft.py"
LAUNCH_SCRIPT="$SCRIPT_DIR/colab_launch_qlora_sft.py"
PACK_SCRIPT="$SCRIPT_DIR/colab_pack_hparam_results.py"
DATA_DIR="$PROJECT_ROOT/data/finetuning/qwen3_hparam_mschema_v1"
AUTH_MODE="adc"
LABEL=""
TRAINING_CONFIG=""
SESSION_NAME=""
RUN_DIR=""
SESSION_ACTIVE=0
RESULTS_COLLECTED=0
ASSUME_YES=0
MAX_STEPS=""

usage() {
  cat <<'EOF'
Run one compact Qwen3 QLoRA hyperparameter trial on a Colab L4.

Usage:
  bash src/scripts/run_colab_qlora_hparam.sh --label LABEL --training-config FILE [options]

Options:
  --data-dir PATH   Database-disjoint HPO package.
  --session NAME    Colab session name; defaults to a unique label-based name.
  --auth MODE       adc or oauth2. Default: adc.
  --max-steps N     Optional optimizer-step cap for a throughput pilot.
  --yes             Confirm compute allocation non-interactively.

Only the final adapter, configs, metrics, and logs are downloaded. The pinned
base model is fetched directly by the remote Colab session and is never stored
on the local machine.
EOF
}

info() { printf '\033[0;34mINFO:\033[0m [%s] %s\n' "${LABEL:-hparam}" "$*"; }
warn() { printf '\033[0;33mWARNING:\033[0m [%s] %s\n' "${LABEL:-hparam}" "$*" >&2; }
die() { printf '\033[0;31mERROR:\033[0m [%s] %s\n' "${LABEL:-hparam}" "$*" >&2; exit 1; }
colab_cmd() { "$COLAB_BIN" --auth="$AUTH_MODE" "$@"; }

collect_results() {
  if (( ! SESSION_ACTIVE || RESULTS_COLLECTED )); then return 0; fi
  local remote_archive="/content/text2sql_sft/hparam-results-transfer.tar.gz"
  local local_archive="$RUN_DIR/hparam-results-transfer.tar.gz"
  if colab_cmd exec -s "$SESSION_NAME" -f "$PACK_SCRIPT" --timeout 900 >>"$RUN_DIR/orchestrator.log" 2>&1 && \
     colab_cmd download -s "$SESSION_NAME" "$remote_archive" "$local_archive" >>"$RUN_DIR/orchestrator.log" 2>&1; then
    mkdir -p "$RUN_DIR/downloaded"
    tar -xzf "$local_archive" -C "$RUN_DIR/downloaded"
    rm -f "$local_archive"
    RESULTS_COLLECTED=1
    info "Compact adapter and metrics downloaded."
    return 0
  fi
  warn "Compact results were unavailable; preserving remote status/log if possible."
  return 1
}

stop_session() {
  if (( SESSION_ACTIVE )); then
    if colab_cmd stop -s "$SESSION_NAME" >>"$RUN_DIR/orchestrator.log" 2>&1; then
      SESSION_ACTIVE=0
    else
      warn "Automatic stop failed: $SESSION_NAME"
    fi
  fi
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] && (( SESSION_ACTIVE )); then
    colab_cmd download -s "$SESSION_NAME" /content/text2sql_sft/status.json "$RUN_DIR/remote-status.json" >/dev/null 2>&1
    colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb" >/dev/null 2>&1
    if (( code == 0 )); then collect_results; fi
    stop_session
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

while (( $# > 0 )); do
  case "$1" in
    --label) [[ $# -ge 2 ]] || die "--label requires a value"; LABEL="$2"; shift 2 ;;
    --training-config) [[ $# -ge 2 ]] || die "--training-config requires a path"; TRAINING_CONFIG="$2"; shift 2 ;;
    --data-dir) [[ $# -ge 2 ]] || die "--data-dir requires a path"; DATA_DIR="$2"; shift 2 ;;
    --session) [[ $# -ge 2 ]] || die "--session requires a value"; SESSION_NAME="$2"; shift 2 ;;
    --auth) [[ $# -ge 2 ]] || die "--auth requires a value"; AUTH_MODE="$2"; shift 2 ;;
    --max-steps) [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--max-steps requires a positive integer"; MAX_STEPS="$2"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -n "$LABEL" ]] || die "--label is required"
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "Invalid label: $LABEL"
[[ -n "$TRAINING_CONFIG" && -f "$TRAINING_CONFIG" ]] || die "Training config is missing"
[[ -d "$DATA_DIR" ]] || die "HPO data package is missing: $DATA_DIR"
[[ -x "$COLAB_BIN" && -x "$BUILD_PYTHON" ]] || die "Required local environments are missing"
case "$AUTH_MODE" in adc|oauth2) ;; *) die "Auth must be adc or oauth2" ;; esac
if (( ! ASSUME_YES )); then
  [[ -t 0 ]] || die "Use --yes for a non-interactive run"
  read -r -p "Create an L4 for trial $LABEL? Type 'yes': " reply
  [[ "$reply" == "yes" ]] || die "Cancelled"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_NAME="${SESSION_NAME:-q3hp-${LABEL}-${STAMP}-$$}"
RUN_DIR="$PROJECT_ROOT/artifacts/qlora-hparam/runs/${STAMP}-${LABEL}-$$"
mkdir -p "$RUN_DIR"
printf '%s\n' "$SESSION_NAME" >"$RUN_DIR/session-name.txt"
printf '%s\n' "$LABEL" >"$RUN_DIR/trial-label.txt"
cp "$TRAINING_CONFIG" "$RUN_DIR/training-config.json"

BUNDLE="$RUN_DIR/sft_bundle.tar.gz"
BUILD_ARGS=(
  --output "$BUNDLE"
  --model qwen3-4b-instruct-2507
  --dataset-variant base
  --data-dir "$DATA_DIR"
  --training-config "$TRAINING_CONFIG"
  --run-name "qwen3-hparam-${LABEL}-${STAMP}"
  --no-resume-smoke-test
)
if [[ -n "$MAX_STEPS" ]]; then BUILD_ARGS+=(--max-steps "$MAX_STEPS"); fi
"$BUILD_PYTHON" "$SCRIPT_DIR/build_colab_sft_bundle.py" "${BUILD_ARGS[@]}" >"$RUN_DIR/bundle-info.json"

info "Creating Colab L4 session $SESSION_NAME"
SESSION_ACTIVE=1
colab_cmd new -s "$SESSION_NAME" --gpu L4 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
colab_cmd status -s "$SESSION_NAME" | tee "$RUN_DIR/session-status.txt" | tee -a "$RUN_DIR/orchestrator.log"

attempt=1
until colab_cmd install -s "$SESSION_NAME" -r "$REMOTE_REQUIREMENTS" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; do
  (( attempt < 3 )) || die "Dependency installation failed three times"
  attempt=$((attempt + 1))
  sleep 10
done

colab_cmd upload -s "$SESSION_NAME" "$BUNDLE" /content/text2sql_sft_bundle.tar.gz 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
rm -f "$BUNDLE"
colab_cmd exec -s "$SESSION_NAME" -f "$PREPARE_SCRIPT" --timeout 900 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
info "Training; Qwen3 base weights are downloading directly inside Colab."
colab_cmd exec -s "$SESSION_NAME" -f "$LAUNCH_SCRIPT" --timeout 43200 2>&1 | tee -a "$RUN_DIR/orchestrator.log"

collect_results
colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb" >/dev/null 2>&1 || true
stop_session
info "Trial complete: $RUN_DIR"
