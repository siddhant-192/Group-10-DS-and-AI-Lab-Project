#!/usr/bin/env python3
"""Fail unless a downloaded Colab evaluation status is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("status", type=Path)
args = parser.parse_args()

payload = json.loads(args.status.read_text(encoding="utf-8"))
phase = payload.get("phase")
if phase != "complete":
    detail = payload.get("error") or "evaluation did not reach the complete phase"
    raise SystemExit(f"Evaluation status is {phase!r}: {detail}")

completed_models = int(payload.get("completed_models", 0))
model_count = int(payload.get("model_count", len(payload.get("models") or [])))
if completed_models != model_count or model_count < 1:
    raise SystemExit(
        f"Evaluation model count mismatch: completed={completed_models}, expected={model_count}"
    )

print(
    "EVAL_STATUS_VALIDATED="
    + json.dumps(
        {
            "phase": phase,
            "completed_models": completed_models,
            "model_count": model_count,
        },
        sort_keys=True,
    )
)
