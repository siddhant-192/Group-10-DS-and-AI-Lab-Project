#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COLAB_BIN="$PROJECT_ROOT/.venv-colab-cli/bin/colab"
BUILD_PYTHON="$PROJECT_ROOT/.venv-model-eval/bin/python"
MODEL_MANIFEST="$PROJECT_ROOT/models/text2sql-eval/download_manifest.json"
MODEL_CONFIG="$PROJECT_ROOT/configs/text2sql_eval_models.json"
DATA_PATH="$PROJECT_ROOT/data/processed/spider/validation.jsonl"
REMOTE_REQUIREMENTS="$SCRIPT_DIR/colab-eval-requirements.txt"
PREPARE_SCRIPT="$SCRIPT_DIR/colab_prepare_zero_shot_eval.py"
LAUNCH_SCRIPT="$SCRIPT_DIR/colab_launch_zero_shot_eval.py"
PACK_SCRIPT="$SCRIPT_DIR/colab_pack_eval_results.py"
STATUS_VALIDATOR="$SCRIPT_DIR/validate_eval_status.py"
PREDICTION_VALIDATOR="$SCRIPT_DIR/validate_prediction_jsonl.py"

AUTH_MODE="adc"
ENGINE="transformers"
SESSION_NAME="text2sql-zero-shot-$(date +%Y%m%d-%H%M%S)-$$"
ASSUME_YES=0
KEEP_SESSION=0
SESSION_ACTIVE=0
RESULTS_COLLECTED=0
LIMIT=""
BATCH_SIZE=""
NUM_CANDIDATES="1"
MAX_NEW_TOKENS="256"
TEMPERATURE="0.7"
TOP_P="0.95"
CANDIDATE_SELECTION="execution-consensus"
SELECTED_MODELS=()
ADAPTER_DIR=""
ADAPTER_LABEL=""
RESUME_PREDICTIONS=""
RUN_DIR=""
BUNDLE_PART_DIR=""

usage() {
  cat <<'EOF'
Run the three pinned base models sequentially on Spider validation using one L4.

Usage:
  bash scripts/run_colab_zero_shot_eval.sh [options]

Options:
  --session NAME       Override the generated Colab session name.
  --auth MODE          adc (recommended) or oauth2. Default: adc.
  --engine ENGINE      transformers (default) or vllm (FINER n-sampling).
  --model SLUG         Evaluate only this configured model; may be repeated.
  --config PATH        Alternate evaluator model configuration.
  --manifest PATH      Alternate pinned revision manifest.
  --data PATH          Alternate aligned evaluation JSONL and referenced databases.
  --limit N            Evaluate only the first N examples (useful for a pilot).
  --batch-size N       Override each model's tested L4 batch-size starting point.
  --num-candidates N   Generate N sampled candidates and select by execution consensus.
  --max-new-tokens N   Maximum generated tokens per candidate. Default: 256.
  --temperature FLOAT  Sampling temperature when N > 1. Default: 0.7.
  --top-p FLOAT        Nucleus-sampling threshold when N > 1. Default: 0.95.
  --candidate-selection MODE
                       execution-consensus or value-aware-voting.
  --adapter-dir PATH   Upload and evaluate one local PEFT adapter.
  --adapter-label NAME Result slug for that adapter; required with --adapter-dir.
  --resume-predictions PATH
                       Seed one model's saved predictions and run only missing IDs.
  --keep-session       Do not stop the L4 automatically after results download.
  --yes                Skip the compute-allocation confirmation.
  -h, --help           Show this help.

The local model snapshots must already exist. The Colab runtime downloads the
same pinned public revisions directly from Hugging Face because Colab CLI file
upload base64-encodes whole files; relaying 15+ GiB of weights through it is much
slower and less reliable than a direct runtime download. Code, validation data,
SQLite databases, status, logs, and results still use Colab CLI upload/download.

Examples:
  # Recommended 12-example end-to-end pilot.
  bash scripts/run_colab_zero_shot_eval.sh --limit 12

  # Full 1,034-example, three-model evaluation.
  bash scripts/run_colab_zero_shot_eval.sh
EOF
}

info() {
  printf '\033[0;34mINFO:\033[0m %s\n' "$*"
}

warn() {
  printf '\033[0;33mWARNING:\033[0m %s\n' "$*" >&2
}

die() {
  printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2
  exit 1
}

confirm() {
  local reply=""
  if (( ASSUME_YES )); then
    return 0
  fi
  [[ -t 0 ]] || die "Confirmation requires an interactive terminal; rerun with --yes."
  read -r -p "Create an L4 session and consume Colab compute? Type 'yes' to continue: " reply
  [[ "$reply" == "yes" ]] || die "Cancelled."
}

colab_cmd() {
  "$COLAB_BIN" --auth="$AUTH_MODE" "$@"
}

upload_with_retry() {
  local source="$1"
  local destination="$2"
  local attempt=1
  while (( attempt <= 3 )); do
    if colab_cmd upload -s "$SESSION_NAME" "$source" "$destination" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; then
      return 0
    fi
    warn "Upload attempt $attempt failed for $(basename "$source")."
    attempt=$((attempt + 1))
    sleep 5
  done
  return 1
}

install_with_retry() {
  local attempt=1
  while (( attempt <= 3 )); do
    if colab_cmd install -s "$SESSION_NAME" -r "$REMOTE_REQUIREMENTS" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; then
      return 0
    fi
    warn "Dependency installation attempt $attempt failed."
    attempt=$((attempt + 1))
    sleep 10
  done
  return 1
}

stop_session() {
  if (( SESSION_ACTIVE )) && (( ! KEEP_SESSION )); then
    info "Stopping L4 session '$SESSION_NAME'."
    if colab_cmd stop -s "$SESSION_NAME"; then
      SESSION_ACTIVE=0
    else
      warn "Automatic stop failed. Run: $COLAB_BIN --auth=$AUTH_MODE stop -s $SESSION_NAME"
      return 1
    fi
  fi
}

collect_results() {
  local remote_archive="/content/text2sql_eval/results-transfer.tar.gz"
  local local_archive="$RUN_DIR/results-transfer.tar.gz"
  if (( ! SESSION_ACTIVE )) || (( RESULTS_COLLECTED )); then
    return 0
  fi
  info "Packaging and downloading completed or partial results."
  if colab_cmd exec -s "$SESSION_NAME" -f "$PACK_SCRIPT" --timeout 600 >>"$RUN_DIR/orchestrator.log" 2>&1 && \
    colab_cmd download -s "$SESSION_NAME" "$remote_archive" "$local_archive" >>"$RUN_DIR/orchestrator.log" 2>&1; then
    mkdir -p "$RUN_DIR/downloaded"
    tar -xzf "$local_archive" -C "$RUN_DIR/downloaded"
    RESULTS_COLLECTED=1
    info "Results downloaded to $RUN_DIR/downloaded"
  else
    warn "Could not download a result archive; inspect $RUN_DIR/orchestrator.log"
    return 1
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  collect_results
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] && (( SESSION_ACTIVE )); then
    colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb" >/dev/null 2>&1
  fi
  stop_session
  if (( KEEP_SESSION )) && (( SESSION_ACTIVE )); then
    warn "Session remains live: $SESSION_NAME"
    warn "Stop it later with: $COLAB_BIN --auth=$AUTH_MODE stop -s $SESSION_NAME"
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

while (( $# > 0 )); do
  case "$1" in
    --session)
      [[ $# -ge 2 ]] || die "--session requires a value."
      SESSION_NAME="$2"
      shift 2
      ;;
    --auth)
      [[ $# -ge 2 ]] || die "--auth requires adc or oauth2."
      case "$2" in
        adc|ADC) AUTH_MODE="adc" ;;
        oauth2|OAUTH2) AUTH_MODE="oauth2" ;;
        *) die "Unsupported auth mode: $2" ;;
      esac
      shift 2
      ;;
    --engine)
      [[ $# -ge 2 ]] || die "--engine requires transformers or vllm."
      case "$2" in
        transformers|vllm) ENGINE="$2" ;;
        *) die "Unsupported inference engine: $2" ;;
      esac
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || die "--model requires a configured slug."
      SELECTED_MODELS+=("$2")
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || die "--config requires a path."
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --manifest)
      [[ $# -ge 2 ]] || die "--manifest requires a path."
      MODEL_MANIFEST="$2"
      shift 2
      ;;
    --data)
      [[ $# -ge 2 ]] || die "--data requires a path."
      DATA_PATH="$2"
      shift 2
      ;;
    --limit)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--limit requires a positive integer."
      LIMIT="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--batch-size requires a positive integer."
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-candidates)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--num-candidates requires a positive integer."
      NUM_CANDIDATES="$2"
      shift 2
      ;;
    --max-new-tokens)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--max-new-tokens requires a positive integer."
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--temperature requires a positive number."
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      [[ $# -ge 2 && "$2" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || die "--top-p must be in (0, 1]."
      TOP_P="$2"
      shift 2
      ;;
    --candidate-selection)
      [[ $# -ge 2 ]] || die "--candidate-selection requires a value."
      case "$2" in
        execution-consensus|value-aware-voting) CANDIDATE_SELECTION="$2" ;;
        *) die "Unsupported candidate selection: $2" ;;
      esac
      shift 2
      ;;
    --adapter-dir)
      [[ $# -ge 2 ]] || die "--adapter-dir requires a path."
      ADAPTER_DIR="$2"
      shift 2
      ;;
    --adapter-label)
      [[ $# -ge 2 ]] || die "--adapter-label requires a value."
      ADAPTER_LABEL="$2"
      shift 2
      ;;
    --resume-predictions)
      [[ $# -ge 2 ]] || die "--resume-predictions requires a path."
      RESUME_PREDICTIONS="$2"
      shift 2
      ;;
    --keep-session)
      KEEP_SESSION=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

if [[ "$ENGINE" == "vllm" ]]; then
  REMOTE_REQUIREMENTS="$SCRIPT_DIR/colab-finer-vllm-requirements.txt"
  LAUNCH_SCRIPT="$SCRIPT_DIR/colab_launch_finer_vllm_eval.py"
  (( ${#SELECTED_MODELS[@]} == 1 )) || die "vLLM evaluation requires exactly one --model."
  [[ "${SELECTED_MODELS[0]}" == "finer-sql-3b-spider" ]] || die "vLLM evaluation currently supports only finer-sql-3b-spider."
fi

[[ "$SESSION_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "Invalid session name: $SESSION_NAME"
[[ -x "$COLAB_BIN" ]] || die "Missing Colab CLI: run scripts/setup_colab_cli.sh first."
[[ -x "$BUILD_PYTHON" ]] || die "Missing model-eval environment: run scripts/download_eval_models.sh first."
[[ -f "$MODEL_MANIFEST" ]] || die "Missing pinned weights/manifest: run scripts/download_eval_models.sh first."
[[ -f "$MODEL_CONFIG" ]] || die "Missing model configuration: $MODEL_CONFIG"
[[ -f "$DATA_PATH" ]] || die "Missing evaluation data: $DATA_PATH"
[[ -f "$REMOTE_REQUIREMENTS" && -f "$PREPARE_SCRIPT" && -f "$LAUNCH_SCRIPT" && -f "$PACK_SCRIPT" && -f "$STATUS_VALIDATOR" && -f "$PREDICTION_VALIDATOR" ]] || die "Evaluation scripts are incomplete."
if [[ -n "$ADAPTER_DIR" ]]; then
  [[ -d "$ADAPTER_DIR" ]] || die "Adapter directory not found: $ADAPTER_DIR"
  [[ -n "$ADAPTER_LABEL" ]] || die "--adapter-label is required with --adapter-dir."
  (( ${#SELECTED_MODELS[@]} == 1 )) || die "Adapter evaluation requires exactly one --model."
elif [[ -n "$ADAPTER_LABEL" ]]; then
  die "--adapter-label requires --adapter-dir."
fi
if [[ -n "$RESUME_PREDICTIONS" ]]; then
  [[ -f "$RESUME_PREDICTIONS" ]] || die "Resume predictions not found: $RESUME_PREDICTIONS"
  (( ${#SELECTED_MODELS[@]} == 1 )) || die "Prediction resume requires exactly one --model."
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${TIMESTAMP}-$$"
RUN_DIR="$PROJECT_ROOT/artifacts/zero-shot-eval/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '%s\n' "$SESSION_NAME" >"$RUN_DIR/session-name.txt"
printf '%s\n' "$AUTH_MODE" >"$RUN_DIR/auth-mode.txt"
printf '%s\n' "$$" >"$RUN_DIR/orchestrator-pid.txt"

BUILD_ARGS=(
  --output "$RUN_DIR/eval_bundle.tar.gz"
  --data "$DATA_PATH"
  --model-config "$MODEL_CONFIG"
  --model-manifest "$MODEL_MANIFEST"
  --model-source huggingface
  --num-candidates "$NUM_CANDIDATES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --candidate-selection "$CANDIDATE_SELECTION"
)
if [[ -n "$LIMIT" ]]; then
  BUILD_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  BUILD_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if (( ${#SELECTED_MODELS[@]} > 0 )); then
  for model in "${SELECTED_MODELS[@]}"; do
    BUILD_ARGS+=(--model "$model")
  done
fi
if [[ -n "$ADAPTER_DIR" ]]; then
  BUILD_ARGS+=(--adapter-dir "$ADAPTER_DIR" --adapter-label "$ADAPTER_LABEL")
fi
if [[ -n "$RESUME_PREDICTIONS" ]]; then
  BUILD_ARGS+=(--resume-predictions "$RESUME_PREDICTIONS")
fi

info "Building the validation/code bundle."
"$BUILD_PYTHON" "$SCRIPT_DIR/build_colab_eval_bundle.py" "${BUILD_ARGS[@]}" | tee "$RUN_DIR/bundle-info.json"
BUNDLE_PART_DIR="$RUN_DIR/bundle-upload-parts"
mkdir -p "$BUNDLE_PART_DIR"
split -b 32m -d -a 3 "$RUN_DIR/eval_bundle.tar.gz" "$BUNDLE_PART_DIR/part-"
info "Run artifacts: $RUN_DIR"
info "Monitor from another terminal: bash scripts/monitor_colab_eval.sh '$RUN_DIR'"
confirm

info "Creating Colab session '$SESSION_NAME' with an L4."
SESSION_ACTIVE=1
if ! colab_cmd new -s "$SESSION_NAME" --gpu L4 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; then
  die "Colab did not create session '$SESSION_NAME'."
fi
colab_cmd status -s "$SESSION_NAME" | tee "$RUN_DIR/session-status.txt" | tee -a "$RUN_DIR/orchestrator.log"

info "Installing pinned inference/evaluation dependencies remotely."
install_with_retry || die "Could not install evaluation dependencies after three attempts."

info "Uploading code, validation data, databases, and any adapter in 32 MiB parts."
upload_with_retry "$RUN_DIR/bundle-info.json" "/content/text2sql_eval_bundle_manifest.json" \
  || die "Could not upload the evaluation bundle manifest."
part_index=0
for part in "$BUNDLE_PART_DIR"/part-*; do
  remote_part="$(printf '/content/text2sql_eval_bundle_part_%03d' "$part_index")"
  upload_with_retry "$part" "$remote_part" || die "Could not upload evaluation bundle part $part_index."
  part_index=$((part_index + 1))
done

info "Verifying the L4 and uploaded bundle."
colab_cmd exec -s "$SESSION_NAME" -f "$PREPARE_SCRIPT" --timeout 600 2>&1 | tee -a "$RUN_DIR/orchestrator.log"

sync_prediction_once() {
  local slug=""
  local recovery_dir="$RUN_DIR/recovery"
  local partial="$recovery_dir/.predictions.partial.jsonl"
  local final="$recovery_dir/predictions-progress.jsonl"
  local count=""
  (( ${#SELECTED_MODELS[@]} == 1 )) || return 0
  slug="${SELECTED_MODELS[0]}"
  mkdir -p "$recovery_dir"
  colab_cmd download -s "$SESSION_NAME" "/content/text2sql_eval/status.json" \
    "$RUN_DIR/remote-status.json" >/dev/null 2>&1 || true
  if ! colab_cmd download -s "$SESSION_NAME" \
      "/content/text2sql_eval/results/$slug/predictions.jsonl" "$partial" >/dev/null 2>&1; then
    return 0
  fi
  if ! count="$("$BUILD_PYTHON" "$PREDICTION_VALIDATOR" "$partial" 2>/dev/null)"; then
    warn "Downloaded prediction checkpoint was incomplete; preserving the prior valid copy."
    return 0
  fi
  mv "$partial" "$final"
  info "Prediction recovery checkpoint verified locally: $count examples."
}

info "Starting sequential zero-shot evaluation. Progress will stream below."
set +e
colab_cmd exec -s "$SESSION_NAME" -f "$LAUNCH_SCRIPT" --timeout 43200 2>&1 \
  | tee -a "$RUN_DIR/orchestrator.log" &
EVAL_EXEC_PID=$!
set -e
while kill -0 "$EVAL_EXEC_PID" >/dev/null 2>&1; do
  sync_prediction_once
  for _poll in {1..12}; do
    kill -0 "$EVAL_EXEC_PID" >/dev/null 2>&1 || break
    sleep 5
  done
done
set +e
wait "$EVAL_EXEC_PID"
EVAL_EXEC_EXIT=$?
set -e
sync_prediction_once
if (( EVAL_EXEC_EXIT != 0 )); then
  die "Remote evaluation ended with status $EVAL_EXEC_EXIT; resume from recovery/predictions-progress.jsonl."
fi
colab_cmd download -s "$SESSION_NAME" "/content/text2sql_eval/status.json" "$RUN_DIR/final-status.json" \
  2>&1 | tee -a "$RUN_DIR/orchestrator.log"
"$BUILD_PYTHON" "$STATUS_VALIDATOR" "$RUN_DIR/final-status.json"

collect_results
colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb"
stop_session

printf '\n\033[1;32mSequential L4 evaluation completed.\033[0m\n'
printf 'Results: %s\n' "$RUN_DIR/downloaded/results"
printf 'Comparison: %s\n' "$RUN_DIR/downloaded/results/comparison.csv"
