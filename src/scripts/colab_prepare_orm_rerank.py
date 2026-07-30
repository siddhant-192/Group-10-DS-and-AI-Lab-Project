#!/usr/bin/env python3
"""Prepare a fresh Colab L4 for ORM candidate reranking."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile

ARCHIVE = Path("/content/orm_rerank_bundle.tar.gz")
ROOT = Path("/content/text2sql_sft")
REQUIRED = (
    ROOT / "output" / "final_adapter" / "adapter_config.json",
    ROOT / "output" / "final_adapter" / "adapter_model.safetensors",
    ROOT / "configs" / "text2sql_eval_models.json",
    ROOT / "models" / "text2sql-eval" / "download_manifest.json",
    Path("/content/evaluate_text2sql_models.py"),
    Path("/content/evaluate_orm_candidate_groups.py"),
    Path("/content/orm_candidate_groups.jsonl"),
)


if not ARCHIVE.is_file():
    raise FileNotFoundError(ARCHIVE)
ROOT.mkdir(parents=True, exist_ok=True)
with tarfile.open(ARCHIVE, "r:gz") as handle:
    handle.extractall(ROOT, filter="data")

missing = [str(path) for path in REQUIRED if not path.is_file()]
if missing:
    raise FileNotFoundError(f"ORM rerank inputs are incomplete: {missing}")

# Colab currently preinstalls torchao 0.10, but PEFT 0.19 treats any installed
# torchao below 0.16 as a hard error even though this ordinary LoRA adapter does
# not use torchao. Removing the unused optional package selects PEFT's standard
# PyTorch linear dispatcher.
uninstall = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    capture_output=True,
    text=True,
)
print(uninstall.stdout.strip())
if uninstall.returncode:
    print(uninstall.stderr.strip())

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

properties = torch.cuda.get_device_properties(0)
print(
    "COLAB_ORM_RERANK_PREPARED="
    + json.dumps(
        {
            "adapter_bytes": (ROOT / "output" / "final_adapter" / "adapter_model.safetensors").stat().st_size,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gib": round(properties.total_memory / (1024**3), 2),
        },
        sort_keys=True,
    )
)
