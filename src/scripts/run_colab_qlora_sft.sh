#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COLAB_BIN="$PROJECT_ROOT/.venv-colab-cli/bin/colab"
BUILD_PYTHON="$PROJECT_ROOT/.venv-model-eval/bin/python"
REMOTE_REQUIREMENTS="$SCRIPT_DIR/colab-sft-requirements.txt"
PREPARE_SCRIPT="$SCRIPT_DIR/colab_prepare_qlora_sft.py"
LAUNCH_SCRIPT="$SCRIPT_DIR/colab_launch_qlora_sft.py"
PACK_SCRIPT="$SCRIPT_DIR/colab_pack_sft_results.py"
CHECKPOINT_INSPECTOR="$SCRIPT_DIR/inspect_sft_checkpoint_archive.py"

AUTH_MODE="adc"
MODEL="qwen3-4b-instruct-2507"
DATASET_VARIANT="curriculum"
DATA_DIR="$PROJECT_ROOT/data/finetuning/spider_sft_v1"
TRAINING_CONFIG="$PROJECT_ROOT/configs/text2sql_qlora_training.json"
SMOKE=0
ASSUME_YES=0
KEEP_SESSION=0
SESSION_ACTIVE=0
RESULTS_COLLECTED=0
MAX_STEPS=""
TRAIN_LIMIT=""
VALIDATION_LIMIT=""
RESUME_SMOKE_TEST=1
RESUME_CHECKPOINT=""
RESUME_PART_DIR=""
RUN_DIR=""
SESSION_NAME="text2sql-qlora-$(date +%Y%m%d-%H%M%S)-$$"

usage() {
  cat <<'EOF'
Train one pinned text-to-SQL model with QLoRA on one Colab L4.

Usage:
  bash scripts/run_colab_qlora_sft.sh [options]

Options:
  --model SLUG             Configured model slug. Default: qwen3-4b-instruct-2507.
  --dataset VARIANT        curriculum or base. Default: curriculum.
  --data-dir PATH          SFT package containing train/validation/manifests.
  --training-config PATH   Override the QLoRA hyperparameter configuration.
  --smoke                  Use 64 train rows, 16 validation rows, and four steps.
                           The smoke run intentionally stops at step 2 and resumes
                           to step 4 to test checkpoint restoration.
  --max-steps N            Override the configured optimizer-step limit.
  --train-limit N          Override the selected training example count.
  --validation-limit N     Override the selected validation example count.
  --no-resume-smoke-test   Run the smoke test as one uninterrupted phase.
  --resume-checkpoint TAR  Resume a full run from a locally exported checkpoint.
  --session NAME           Override the generated Colab session name.
  --auth MODE              adc (recommended) or oauth2. Default: adc.
  --keep-session           Leave the L4 running after artifacts are downloaded.
  --yes                    Skip the compute-allocation confirmation.
  -h, --help               Show this help.

Examples:
  bash scripts/run_colab_qlora_sft.sh --smoke
  bash scripts/run_colab_qlora_sft.sh --model qwen3-4b-instruct-2507 --dataset curriculum
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
  [[ -t 0 ]] || die "Confirmation requires a terminal; rerun with --yes."
  read -r -p "Create an L4 session and consume Colab compute for QLoRA? Type 'yes': " reply
  [[ "$reply" == "yes" ]] || die "Cancelled."
}

colab_cmd() {
  "$COLAB_BIN" --auth="$AUTH_MODE" "$@"
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
  local remote_archive="/content/text2sql_sft/sft-results-transfer.tar.gz"
  local local_archive="$RUN_DIR/sft-results-transfer.tar.gz"
  if (( ! SESSION_ACTIVE )) || (( RESULTS_COLLECTED )); then
    return 0
  fi
  info "Packaging and downloading completed or partial QLoRA artifacts."
  if colab_cmd exec -s "$SESSION_NAME" -f "$PACK_SCRIPT" --timeout 1800 >>"$RUN_DIR/orchestrator.log" 2>&1 && \
    colab_cmd download -s "$SESSION_NAME" "$remote_archive" "$local_archive" >>"$RUN_DIR/orchestrator.log" 2>&1; then
    mkdir -p "$RUN_DIR/downloaded"
    tar -xzf "$local_archive" -C "$RUN_DIR/downloaded"
    RESULTS_COLLECTED=1
    info "Training artifacts downloaded to $RUN_DIR/downloaded"
  else
    warn "Could not download the result archive; inspect $RUN_DIR/orchestrator.log"
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
    --model)
      [[ $# -ge 2 ]] || die "--model requires a slug."
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      [[ $# -ge 2 ]] || die "--dataset requires base or curriculum."
      case "$2" in base|curriculum) DATASET_VARIANT="$2" ;; *) die "Unsupported dataset: $2" ;; esac
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a path."
      DATA_DIR="$2"
      shift 2
      ;;
    --training-config)
      [[ $# -ge 2 ]] || die "--training-config requires a path."
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --max-steps)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--max-steps requires a positive integer."
      MAX_STEPS="$2"
      shift 2
      ;;
    --train-limit)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--train-limit requires a positive integer."
      TRAIN_LIMIT="$2"
      shift 2
      ;;
    --validation-limit)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--validation-limit requires a positive integer."
      VALIDATION_LIMIT="$2"
      shift 2
      ;;
    --no-resume-smoke-test)
      RESUME_SMOKE_TEST=0
      shift
      ;;
    --resume-checkpoint)
      [[ $# -ge 2 ]] || die "--resume-checkpoint requires a tar archive."
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    --session)
      [[ $# -ge 2 ]] || die "--session requires a value."
      SESSION_NAME="$2"
      shift 2
      ;;
    --auth)
      [[ $# -ge 2 ]] || die "--auth requires adc or oauth2."
      case "$2" in adc|ADC) AUTH_MODE="adc" ;; oauth2|OAUTH2) AUTH_MODE="oauth2" ;; *) die "Unsupported auth mode: $2" ;; esac
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

[[ "$SESSION_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "Invalid session name: $SESSION_NAME"
[[ -x "$COLAB_BIN" ]] || die "Missing Colab CLI: run scripts/setup_colab_cli.sh first."
[[ -x "$BUILD_PYTHON" ]] || die "Missing local model environment."
[[ -d "$DATA_DIR" ]] || die "SFT data package not found: $DATA_DIR"
[[ -f "$TRAINING_CONFIG" ]] || die "Training configuration not found: $TRAINING_CONFIG"
[[ -f "$REMOTE_REQUIREMENTS" && -f "$PREPARE_SCRIPT" && -f "$LAUNCH_SCRIPT" && -f "$PACK_SCRIPT" && -f "$CHECKPOINT_INSPECTOR" ]] || die "QLoRA scripts are incomplete."
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  [[ $SMOKE -eq 0 ]] || die "External checkpoint resume is only supported for full runs."
  [[ -f "$RESUME_CHECKPOINT" ]] || die "Resume checkpoint does not exist: $RESUME_CHECKPOINT"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${TIMESTAMP}-${MODEL}-$$"
RUN_DIR="$PROJECT_ROOT/artifacts/qlora-training/runs/$RUN_ID"
RUN_NAME="${MODEL}-${DATASET_VARIANT}-$([[ $SMOKE -eq 1 ]] && printf smoke || printf full)-$TIMESTAMP"
mkdir -p "$RUN_DIR"
printf '%s\n' "$SESSION_NAME" >"$RUN_DIR/session-name.txt"
printf '%s\n' "$AUTH_MODE" >"$RUN_DIR/auth-mode.txt"
printf '%s\n' "$$" >"$RUN_DIR/orchestrator-pid.txt"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  printf '%s\n' "$RESUME_CHECKPOINT" >"$RUN_DIR/resume-checkpoint-source.txt"
  "$BUILD_PYTHON" "$CHECKPOINT_INSPECTOR" validate "$RESUME_CHECKPOINT" >"$RUN_DIR/resume-checkpoint-validation.json"
  RESUME_PART_DIR="$RUN_DIR/resume-upload-parts"
  mkdir -p "$RESUME_PART_DIR"
  split -b 32m -d -a 3 "$RESUME_CHECKPOINT" "$RESUME_PART_DIR/part-"
fi

BUILD_ARGS=(
  --output "$RUN_DIR/sft_bundle.tar.gz"
  --model "$MODEL"
  --dataset-variant "$DATASET_VARIANT"
  --data-dir "$DATA_DIR"
  --training-config "$TRAINING_CONFIG"
  --run-name "$RUN_NAME"
)
if (( SMOKE )); then BUILD_ARGS+=(--smoke); fi
if [[ -n "$MAX_STEPS" ]]; then BUILD_ARGS+=(--max-steps "$MAX_STEPS"); fi
if [[ -n "$TRAIN_LIMIT" ]]; then BUILD_ARGS+=(--train-limit "$TRAIN_LIMIT"); fi
if [[ -n "$VALIDATION_LIMIT" ]]; then BUILD_ARGS+=(--validation-limit "$VALIDATION_LIMIT"); fi
if (( ! RESUME_SMOKE_TEST )); then BUILD_ARGS+=(--no-resume-smoke-test); fi

info "Building the QLoRA code/data bundle."
"$BUILD_PYTHON" "$SCRIPT_DIR/build_colab_sft_bundle.py" "${BUILD_ARGS[@]}" | tee "$RUN_DIR/bundle-info.json"
info "Run artifacts: $RUN_DIR"
info "Monitor separately: bash scripts/monitor_colab_sft.sh '$RUN_DIR'"
confirm

info "Creating Colab session '$SESSION_NAME' with an L4."
SESSION_ACTIVE=1
if ! colab_cmd new -s "$SESSION_NAME" --gpu L4 2>&1 | tee -a "$RUN_DIR/orchestrator.log"; then
  die "Colab did not create session '$SESSION_NAME'."
fi
colab_cmd status -s "$SESSION_NAME" | tee "$RUN_DIR/session-status.txt" | tee -a "$RUN_DIR/orchestrator.log"

info "Installing the pinned QLoRA dependencies remotely."
install_with_retry || die "Could not install QLoRA dependencies after three attempts."

info "Uploading the training code and SFT data."
colab_cmd upload -s "$SESSION_NAME" "$RUN_DIR/sft_bundle.tar.gz" "/content/text2sql_sft_bundle.tar.gz" 2>&1 | tee -a "$RUN_DIR/orchestrator.log"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
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
  info "Uploading the verified resume checkpoint in 32 MiB parts."
  upload_with_retry "$RUN_DIR/resume-checkpoint-validation.json" "/content/text2sql_resume_manifest.json" \
    || die "Could not upload the resume manifest."
  part_index=0
  for part in "$RESUME_PART_DIR"/part-*; do
    remote_part="$(printf '/content/text2sql_resume_part_%03d' "$part_index")"
    upload_with_retry "$part" "$remote_part" || die "Could not upload resume part $part_index."
    part_index=$((part_index + 1))
  done
fi

info "Verifying the L4, dependencies, and uploaded bundle."
colab_cmd exec -s "$SESSION_NAME" -f "$PREPARE_SCRIPT" --timeout 900 2>&1 | tee -a "$RUN_DIR/orchestrator.log"

sync_checkpoint_once() {
  local remote_status="/content/text2sql_sft/status.json"
  local status_copy="$RUN_DIR/remote-status.json"
  local metadata=""
  local step=""
  local remote_archive=""
  local expected_bytes=""
  local expected_sha256=""
  local export_dir="$RUN_DIR/checkpoint-exports"
  local final_archive=""
  local partial_archive=""
  local validation=""
  if ! colab_cmd download -s "$SESSION_NAME" "$remote_status" "$status_copy" >/dev/null 2>&1; then
    return 0
  fi
  if ! metadata="$("$BUILD_PYTHON" "$CHECKPOINT_INSPECTOR" metadata "$status_copy" 2>/dev/null)"; then
    return 0
  fi
  IFS=$'\t' read -r step remote_archive expected_bytes expected_sha256 <<<"$metadata"
  mkdir -p "$export_dir"
  final_archive="$export_dir/checkpoint-$step.tar"
  validation="$export_dir/checkpoint-$step.validation.json"
  if [[ -f "$final_archive" && -f "$validation" ]]; then
    return 0
  fi
  partial_archive="$export_dir/.checkpoint-$step.partial.tar"
  info "Downloading resumable checkpoint $step while training continues."
  if ! colab_cmd download -s "$SESSION_NAME" "$remote_archive" "$partial_archive" >>"$RUN_DIR/orchestrator.log" 2>&1; then
    warn "Checkpoint $step download failed; the next monitor pass will retry."
    return 0
  fi
  if ! "$BUILD_PYTHON" "$CHECKPOINT_INSPECTOR" validate "$partial_archive" \
      --expected-bytes "$expected_bytes" --expected-sha256 "$expected_sha256" >/dev/null; then
    warn "Checkpoint $step failed local validation; preserving the partial file for diagnosis."
    return 0
  fi
  mv "$partial_archive" "$final_archive"
  "$BUILD_PYTHON" "$CHECKPOINT_INSPECTOR" validate "$final_archive" \
    --expected-bytes "$expected_bytes" --expected-sha256 "$expected_sha256" >"$validation"
  printf '%s\n' "$final_archive" >"$RUN_DIR/latest-checkpoint.txt"
  info "Checkpoint $step is verified locally: $final_archive"
}

info "Starting QLoRA training. Progress will stream below."
set +e
colab_cmd exec -s "$SESSION_NAME" -f "$LAUNCH_SCRIPT" --timeout 43200 2>&1 | tee -a "$RUN_DIR/orchestrator.log" &
TRAIN_EXEC_PID=$!
set -e
while kill -0 "$TRAIN_EXEC_PID" >/dev/null 2>&1; do
  sync_checkpoint_once
  sleep 30
done
set +e
wait "$TRAIN_EXEC_PID"
TRAIN_EXEC_EXIT=$?
set -e
sync_checkpoint_once
if (( TRAIN_EXEC_EXIT != 0 )); then
  die "Remote training execution ended with status $TRAIN_EXEC_EXIT. Use the latest verified checkpoint export to resume."
fi

collect_results
colab_cmd log -s "$SESSION_NAME" -o "$RUN_DIR/session-log.ipynb"
stop_session

printf '\n\033[1;32mQLoRA training run completed.\033[0m\n'
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Adapter: %s\n' "$RUN_DIR/downloaded/output/final_adapter"
