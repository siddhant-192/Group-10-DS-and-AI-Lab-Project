#!/usr/bin/env python3
"""Filter JSONL predictions to the IDs present in an aligned validation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ids = {
        str(json.loads(line)["id"])
        for line in args.validation.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    selected = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("id")) in ids:
            selected.append(row)
    found = {str(row.get("id")) for row in selected}
    missing = sorted(ids - found)
    if missing:
        raise RuntimeError(f"missing {len(missing)} validation IDs; first={missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({"validation_ids": len(ids), "predictions_written": len(selected)}))


if __name__ == "__main__":
    main()
