#!/usr/bin/env python3
"""Validate an incrementally downloaded text-to-SQL prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    rows = []
    with args.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not row.get("id") or "predicted_sql" not in row:
                raise ValueError(f"Prediction line {line_number} is missing id or predicted_sql")
            expected = int(row.get("num_candidates") or 1)
            candidates = row.get("candidates")
            if expected > 1 and (not isinstance(candidates, list) or len(candidates) != expected):
                raise ValueError(
                    f"Prediction line {line_number} expected {expected} complete candidates"
                )
            rows.append(row)
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Prediction JSONL is empty or contains duplicate IDs")
    print(len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
