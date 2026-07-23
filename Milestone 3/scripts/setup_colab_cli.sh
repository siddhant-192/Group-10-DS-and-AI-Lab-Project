#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-colab-cli"
REMOTE_REQUIREMENTS="$SCRIPT_DIR/colab-remote-requirements.txt"
SMOKE_SCRIPT="$SCRIPT_DIR/colab_smoke_test.py"

AUTH_MODE="adc"
GPU=""
SESSION_NAME="text2sql-smoke-$(date +%Y%m%d-%H%M%S)-$$"
INSTALL_REMOTE_DEPS=1
FORCE_REAUTH=0
ASSUME_YES=0
SESSION_ACTIVE=0
TEMP_DIR=""
ARTIFACT_DIR=""
COLAB_BIN=""

ADC_SCOPES="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"

usage() {
  cat <<'EOF'
Set up the Google Colab CLI and run an end-to-end smoke test.

Usage:
  bash scripts/setup_colab_cli.sh [options]

Options:
  --gpu TYPE            Request T4, L4, G4, A100, or H100. Default: CPU.
  --session NAME        Override the generated Colab session name.
  --auth MODE           Use adc (recommended) or oauth2. Default: adc.
  --reauth              Force a fresh ADC browser login.
  --skip-remote-deps    Skip installing the text-to-SQL training packages.
  --yes                 Accept prerequisite-install and compute prompts.
  -h, --help            Show this help.

Examples:
  # Safest first run: CPU, ADC login, remote packages, complete smoke test.
  bash scripts/setup_colab_cli.sh

  # Verify that a T4 is actually available.
  bash scripts/setup_colab_cli.sh --gpu T4

  # Avoid installing Google Cloud CLI by using Colab's copy/paste OAuth flow.
  bash scripts/setup_colab_cli.sh --auth oauth2

The script always stops the Colab session, including after an error or Ctrl-C.
It never mounts Google Drive. Local scripts are transmitted by `colab exec`, and
the tiny upload/download test uses the runtime's ephemeral /content directory.
EOF
}

step() {
  printf '\n\033[1;36m[%s/9] %s\033[0m\n' "$1" "$2"
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
  local prompt="$1"
  local reply=""

  if (( ASSUME_YES )); then
    info "$prompt (accepted by --yes)"
    return 0
  fi

  if [[ ! -t 0 ]]; then
    die "A confirmation is required but stdin is not interactive. Re-run in a terminal or pass --yes."
  fi

  read -r -p "$prompt Type 'yes' to continue: " reply
  [[ "$reply" == "yes" ]] || die "Cancelled."
}

colab_cmd() {
  "$COLAB_BIN" --auth="$AUTH_MODE" "$@"
}

find_compatible_python() {
  local candidate=""

  for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

install_python_with_homebrew() {
  command -v brew >/dev/null 2>&1 || die \
    "Python 3.12+ is required. Install it from https://www.python.org/downloads/ and re-run this script."

  confirm "Python 3.12+ was not found. Install python@3.12 with Homebrew?"
  brew install python@3.12
}

find_gcloud() {
  local brew_prefix=""
  local candidate=""

  if command -v gcloud >/dev/null 2>&1; then
    command -v gcloud
    return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    brew_prefix="$(brew --prefix)"
    for candidate in \
      "$brew_prefix/bin/gcloud" \
      "$brew_prefix/share/google-cloud-sdk/bin/gcloud" \
      "$brew_prefix/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud"; do
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi

  return 1
}

install_gcloud_with_homebrew() {
  command -v brew >/dev/null 2>&1 || die \
    "Google Cloud CLI is required for --auth adc. Install it from https://cloud.google.com/sdk/docs/install or use --auth oauth2."

  confirm "Google Cloud CLI was not found. Install google-cloud-sdk with Homebrew?"
  brew install --cask google-cloud-sdk
}

stop_session() {
  if (( ! SESSION_ACTIVE )); then
    return 0
  fi

  info "Stopping Colab session '$SESSION_NAME' so it cannot keep consuming compute."
  if colab_cmd stop -s "$SESSION_NAME"; then
    SESSION_ACTIVE=0
    return 0
  fi

  warn "Automatic cleanup failed. Run this immediately:"
  warn "$COLAB_BIN --auth=$AUTH_MODE stop -s $SESSION_NAME"
  return 1
}

cleanup() {
  local exit_code=$?
  local cleanup_failed=0

  trap - EXIT INT TERM
  set +e

  if ! stop_session; then
    cleanup_failed=1
  fi

  if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
    rm -f "$TEMP_DIR/source.txt" "$TEMP_DIR/downloaded.txt"
    rmdir "$TEMP_DIR" 2>/dev/null || true
  fi

  if (( exit_code == 0 && cleanup_failed )); then
    exit_code=1
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

while (( $# > 0 )); do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || die "--gpu requires a value."
      case "$2" in
        t4|T4) GPU="T4" ;;
        l4|L4) GPU="L4" ;;
        g4|G4) GPU="G4" ;;
        a100|A100) GPU="A100" ;;
        h100|H100) GPU="H100" ;;
        *) die "Unsupported GPU '$2'. Choose T4, L4, G4, A100, or H100." ;;
      esac
      shift 2
      ;;
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
        *) die "Unsupported auth mode '$2'. Choose adc or oauth2." ;;
      esac
      shift 2
      ;;
    --reauth)
      FORCE_REAUTH=1
      shift
      ;;
    --skip-remote-deps)
      INSTALL_REMOTE_DEPS=0
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
      die "Unknown option: $1. Run with --help for usage."
      ;;
  esac
done

[[ "$AUTH_MODE" == "adc" || "$AUTH_MODE" == "oauth2" ]] || die \
  "Unsupported auth mode '$AUTH_MODE'. Choose adc or oauth2."

if [[ -n "$GPU" ]]; then
  case "$GPU" in
    T4|L4|G4|A100|H100) ;;
    *) die "Unsupported GPU '$GPU'. Choose T4, L4, G4, A100, or H100." ;;
  esac
fi

[[ "$SESSION_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die \
  "Session names may contain only letters, numbers, underscores, and hyphens, and must start with a letter or number."

[[ -f "$REMOTE_REQUIREMENTS" ]] || die "Missing $REMOTE_REQUIREMENTS"
[[ -f "$SMOKE_SCRIPT" ]] || die "Missing $SMOKE_SCRIPT"

step 1 "Checking the operating system and local prerequisites"
case "$(uname -s)" in
  Darwin|Linux) ;;
  *) die "The official Colab CLI currently supports macOS and Linux only." ;;
esac

PYTHON_BIN="$(find_compatible_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  install_python_with_homebrew
  PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"
fi
[[ -x "$PYTHON_BIN" ]] || die "A compatible Python executable was not found at $PYTHON_BIN"
info "Using $("$PYTHON_BIN" --version 2>&1) at $PYTHON_BIN"

step 2 "Creating or refreshing the local Colab CLI virtual environment"
if [[ -x "$VENV_DIR/bin/python" ]] && ! \
  "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1; then
  warn "$VENV_DIR uses Python older than 3.12. Move or remove it, then re-run the script."
  die "The existing virtual environment is incompatible with google-colab-cli."
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade google-colab-cli
COLAB_BIN="$VENV_DIR/bin/colab"
[[ -x "$COLAB_BIN" ]] || die "Colab CLI installation did not create $COLAB_BIN"
info "Installed $("$COLAB_BIN" version)"

step 3 "Authenticating the local CLI with Google"
if [[ "$AUTH_MODE" == "adc" ]]; then
  GCLOUD_BIN="$(find_gcloud || true)"
  if [[ -z "$GCLOUD_BIN" ]]; then
    install_gcloud_with_homebrew
    GCLOUD_BIN="$(find_gcloud || true)"
  fi
  [[ -n "$GCLOUD_BIN" ]] || die \
    "Google Cloud CLI installation completed but gcloud is not on PATH. Open a new terminal and re-run, or use --auth oauth2."

  if (( FORCE_REAUTH )) || ! colab_cmd sessions >/dev/null 2>&1; then
    info "A browser will open for Google login. Approve the requested Colab scopes."
    "$GCLOUD_BIN" auth application-default login --scopes="$ADC_SCOPES"
  fi
else
  if (( FORCE_REAUTH )); then
    warn "--reauth applies to ADC. OAuth2 refresh tokens are managed by the Colab CLI."
  fi
  info "On first use, open the printed URL, sign in, then paste the authorization code here."
fi

colab_cmd sessions
info "CLI authentication succeeded."

step 4 "Reviewing the temporary compute allocation"
if [[ -n "$GPU" ]]; then
  info "Requested accelerator: $GPU GPU"
else
  info "Requested accelerator: CPU"
fi
warn "A live CLI session uses your Colab allocation and may consume paid compute units until stopped."
confirm "Create the temporary session '$SESSION_NAME'?"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARTIFACT_DIR="$PROJECT_ROOT/artifacts/colab-smoke/$TIMESTAMP"
mkdir -p "$ARTIFACT_DIR"

step 5 "Provisioning the named Colab session"
if [[ -n "$GPU" ]]; then
  colab_cmd new -s "$SESSION_NAME" --gpu "$GPU"
else
  colab_cmd new -s "$SESSION_NAME"
fi
SESSION_ACTIVE=1
colab_cmd status -s "$SESSION_NAME" | tee "$ARTIFACT_DIR/session-status.txt"

step 6 "Installing the remote text-to-SQL training dependencies"
if (( INSTALL_REMOTE_DEPS )); then
  colab_cmd install -s "$SESSION_NAME" -r "$REMOTE_REQUIREMENTS"
else
  info "Skipped by --skip-remote-deps; the base Colab torch installation will still be tested."
fi

step 7 "Executing the Python and accelerator smoke test remotely"
colab_cmd exec -s "$SESSION_NAME" -f "$SMOKE_SCRIPT" | tee "$ARTIFACT_DIR/remote-smoke.txt"
grep -q 'COLAB_SMOKE_JSON=' "$ARTIFACT_DIR/remote-smoke.txt" || die \
  "The remote script returned without the expected smoke-test marker."

if [[ -n "$GPU" ]] && ! grep -Eq '"cuda_available"[[:space:]]*:[[:space:]]*true' "$ARTIFACT_DIR/remote-smoke.txt"; then
  die "A $GPU was requested, but torch.cuda.is_available() was not true."
fi

step 8 "Testing upload and download without Google Drive"
TEMP_DIR="$(mktemp -d)"
printf 'colab-cli-transfer-ok session=%s\n' "$SESSION_NAME" > "$TEMP_DIR/source.txt"
REMOTE_TEST_FILE="/content/colab-cli-transfer-$SESSION_NAME.txt"

colab_cmd upload -s "$SESSION_NAME" "$TEMP_DIR/source.txt" "$REMOTE_TEST_FILE"
colab_cmd ls -s "$SESSION_NAME" "$REMOTE_TEST_FILE"
colab_cmd download -s "$SESSION_NAME" "$REMOTE_TEST_FILE" "$TEMP_DIR/downloaded.txt"
cmp -s "$TEMP_DIR/source.txt" "$TEMP_DIR/downloaded.txt" || die \
  "The downloaded file did not match the uploaded file."
colab_cmd rm -s "$SESSION_NAME" "$REMOTE_TEST_FILE"
info "Round-trip file transfer succeeded."

step 9 "Exporting the replayable log and releasing compute"
colab_cmd log -s "$SESSION_NAME" -o "$ARTIFACT_DIR/session-log.ipynb"
stop_session

printf '\n\033[1;32mColab CLI setup and smoke test passed.\033[0m\n'
printf 'Local CLI environment: %s\n' "$VENV_DIR"
printf 'Smoke-test artifacts:  %s\n' "$ARTIFACT_DIR"
printf 'Activate manually with: source %q\n' "$VENV_DIR/bin/activate"
