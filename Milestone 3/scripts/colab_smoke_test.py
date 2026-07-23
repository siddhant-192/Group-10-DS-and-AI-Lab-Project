"""Minimal remote-runtime smoke test for scripts/setup_colab_cli.sh."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys


PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "evaluate",
    "peft",
    "safetensors",
    "sentencepiece",
    "sqlglot",
    "transformers",
    "trl",
)


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    import torch

    cuda_available = torch.cuda.is_available()
    disk_total, disk_used, disk_free = shutil.disk_usage("/content")

    result = {
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_count": torch.cuda.device_count(),
        "cwd": os.getcwd(),
        "disk_free_gib": round(disk_free / 1024**3, 2),
        "disk_total_gib": round(disk_total / 1024**3, 2),
        "disk_used_gib": round(disk_used / 1024**3, 2),
        "packages": {package: installed_version(package) for package in PACKAGES},
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }

    print("COLAB_SMOKE_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
