#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

failures=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

printf '[1/7] High-confidence credential scan\n'
if rg -n --hidden \
  -g '!.git/**' \
  -g '!scripts/audit_public_package.sh' \
  '(hf_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer[[:space:]]+[A-Za-z0-9._-]{20,})' .; then
  printf 'ERROR: possible credential found.\n' >&2
  failures=1
fi

printf '[2/7] Private and machine-specific path scan\n'
if rg -n --hidden \
  -g '!.git/**' \
  -g '!scripts/audit_public_package.sh' \
  '(/Users/|/home/[^ /]+/|/private/tmp|IITM Email Drive:|phoenix@)' .; then
  printf 'ERROR: private or machine-specific path found.\n' >&2
  failures=1
fi

printf '[3/7] Large and prohibited file scan\n'
large_files="$(find . -type f -size +95M -not -path './.git/*' -print)"
if [[ -n "$large_files" ]]; then
  printf '%s\n' "$large_files"
  printf 'ERROR: file larger than 95 MiB found.\n' >&2
  failures=1
fi
if find . -type f \
  \( -name '*.safetensors' -o -name '*.sqlite' -o -name '*.parquet' \
     -o -name '*.pem' -o -name 'client_secret*.json' \) \
  -not -path './.git/*' -print | grep -q .; then
  find . -type f \
    \( -name '*.safetensors' -o -name '*.sqlite' -o -name '*.parquet' \
       -o -name '*.pem' -o -name 'client_secret*.json' \) \
    -not -path './.git/*' -print
  printf 'ERROR: prohibited binary or credential file found.\n' >&2
  failures=1
fi

printf '[4/7] JSON parse check\n'
while IFS= read -r -d '' file; do
  "$PYTHON_BIN" -m json.tool "$file" >/dev/null
done < <(find . -type f -name '*.json' -not -path './.git/*' -print0)

printf '[5/7] Python compilation\n'
"$PYTHON_BIN" -m compileall -q src scripts tests

printf '[6/7] Shell syntax\n'
while IFS= read -r -d '' file; do
  bash -n "$file"
done < <(find scripts -type f -name '*.sh' -print0)

printf '[7/7] Unit tests\n'
"$PYTHON_BIN" -m pytest -q

if [[ "$failures" -ne 0 ]]; then
  printf 'Public-package audit FAILED.\n' >&2
  exit 1
fi

printf 'Public-package audit passed.\n'
