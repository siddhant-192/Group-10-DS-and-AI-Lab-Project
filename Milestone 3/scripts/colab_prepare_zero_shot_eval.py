"""Extract and verify /content/text2sql_eval_bundle.tar.gz on Colab."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import torch


BUNDLE = Path("/content/text2sql_eval_bundle.tar.gz")
BUNDLE_MANIFEST = Path("/content/text2sql_eval_bundle_manifest.json")
ROOT = Path("/content/text2sql_eval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def version_pair(value: str) -> tuple[int, int]:
    numbers = []
    for component in value.split(".")[:2]:
        digits = "".join(character for character in component if character.isdigit())
        numbers.append(int(digits or 0))
    return tuple((numbers + [0, 0])[:2])


if not torch.cuda.is_available():
    raise RuntimeError("The requested Colab runtime does not expose CUDA")
gpu_name = torch.cuda.get_device_name(0)
if "L4" not in gpu_name.upper():
    raise RuntimeError(f"Expected an L4 runtime, but Colab allocated: {gpu_name}")

bundle_parts = sorted(Path("/content").glob("text2sql_eval_bundle_part_*"))
if bundle_parts:
    if not BUNDLE_MANIFEST.exists():
        raise FileNotFoundError(BUNDLE_MANIFEST)
    expected_bundle = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    with BUNDLE.open("wb") as destination:
        for part in bundle_parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if BUNDLE.stat().st_size != int(expected_bundle["bytes"]):
        raise RuntimeError("Reassembled evaluation bundle size mismatch")
    if sha256(BUNDLE) != str(expected_bundle["sha256"]):
        raise RuntimeError("Reassembled evaluation bundle checksum mismatch")
elif not BUNDLE.exists():
    raise FileNotFoundError(BUNDLE)

torchao_action = "not-installed"
try:
    torchao_version = importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    if version_pair(torchao_version) < (0, 16):
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            check=True,
        )
        importlib.invalidate_caches()
        torchao_action = f"removed-incompatible-{torchao_version}"
    else:
        torchao_action = f"kept-{torchao_version}"

ROOT.mkdir(parents=True, exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as archive:
    root = ROOT.resolve()
    for member in archive.getmembers():
        target = (ROOT / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Unsafe bundle member: {member.name}")
    archive.extractall(ROOT, filter="data")

manifest = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = ROOT / item["path"]
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"Bundle verification failed: {item['path']}")

disk = shutil.disk_usage("/content")
print(
    "COLAB_EVAL_PREPARED="
    + json.dumps(
        {
            "gpu": gpu_name,
            "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "disk_free_gib": round(disk.free / 1024**3, 2),
            "validation_examples": manifest["validation_examples"],
            "validation_databases": manifest["validation_databases"],
            "bundle_parts": len(bundle_parts),
            "torchao": torchao_action,
        },
        sort_keys=True,
    )
)
