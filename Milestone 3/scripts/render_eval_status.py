#!/usr/bin/env python3
"""Render the Colab evaluator status JSON as one compact terminal update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("status", type=Path)
args = parser.parse_args()
payload = json.loads(args.status.read_text(encoding="utf-8"))
age = max(0, int(time.time() - float(payload.get("updated_at_epoch", time.time()))))
cuda = payload.get("cuda") or {}
parts = [
    f"phase={payload.get('phase', 'unknown')}",
    f"model={payload.get('current_model') or '-'}",
    f"models={payload.get('model_index', 0)}/{payload.get('model_count', 0)}",
    f"examples={payload.get('completed_examples', 0)}/{payload.get('total_examples', 0)}",
    f"batch={payload.get('batch_size', '-')}",
    f"VRAM={cuda.get('allocated_gib', 0):.2f}/{payload.get('gpu_total_gib', 0):.2f}GiB",
    f"heartbeat={age}s ago",
]
if payload.get("error"):
    parts.append(f"error={payload['error']}")
print(" | ".join(parts))
