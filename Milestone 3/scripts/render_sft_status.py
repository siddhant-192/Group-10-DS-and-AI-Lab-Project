#!/usr/bin/env python3
"""Render one compact Colab QLoRA status line."""

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
loss = payload.get("loss", payload.get("eval_loss"))
parts = [
    f"phase={payload.get('phase', 'unknown')}",
    f"stage={payload.get('phase_label', '-')}",
    f"model={payload.get('model', '-')}",
    f"data={payload.get('dataset_variant', '-')}",
    f"step={payload.get('step', 0)}/{payload.get('max_steps', '-')}",
    f"epoch={float(payload.get('epoch') or 0):.3f}",
    f"loss={float(loss):.4f}" if loss is not None else "loss=-",
    f"VRAM={cuda.get('allocated_gib', 0):.2f}/{cuda.get('max_allocated_gib', 0):.2f}GiB",
    f"heartbeat={age}s",
]
if payload.get("error"):
    parts.append(f"error={payload['error']}")
print(" | ".join(parts))
