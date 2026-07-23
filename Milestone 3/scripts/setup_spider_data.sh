#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-data"
REQUIREMENTS="$PROJECT_ROOT/requirements-data.txt"

find_python() {
  local candidate=""
  for candidate in python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  printf 'ERROR: Python 3.10+ is required.\n' >&2
  exit 1
fi

printf '\n[1/4] Creating the dedicated data environment\n'
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

printf '\n[2/4] Installing the data-pipeline requirements\n'
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"

printf '\n[3/4] Downloading and validating Spider annotations\n'
"$VENV_DIR/bin/python" "$SCRIPT_DIR/prepare_spider_data.py" download

printf '\n[4/4] Building schemas, validating SQL, and producing JSONL + EDA\n'
"$VENV_DIR/bin/python" "$SCRIPT_DIR/prepare_spider_data.py" prepare "$@"

printf '\nData pipeline complete. Outputs: %s\n' "$PROJECT_ROOT/data/processed/spider"

