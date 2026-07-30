"""Extract, verify, and hardware-check a Colab QLoRA training bundle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tarfile

import bitsandbytes
import torch


BUNDLE = Path("/content/text2sql_sft_bundle.tar.gz")
RESUME_ARCHIVE = Path("/content/text2sql_resume_checkpoint.tar")
RESUME_MANIFEST = Path("/content/text2sql_resume_manifest.json")
RESUME_PARTS = "/content/text2sql_resume_part_*"
ROOT = Path("/content/text2sql_sft")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if not torch.cuda.is_available():
    raise RuntimeError("The requested Colab runtime does not expose CUDA")
gpu_name = torch.cuda.get_device_name(0)
if "L4" not in gpu_name.upper():
    raise RuntimeError(f"Expected an NVIDIA L4, but Colab allocated: {gpu_name}")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The L4 runtime does not report bfloat16 support")
if not BUNDLE.exists():
    raise FileNotFoundError(BUNDLE)

ROOT.mkdir(parents=True, exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as archive:
    root = ROOT.resolve()
    for member in archive.getmembers():
        target = (ROOT / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Unsafe bundle member: {member.name}")
    archive.extractall(ROOT, filter="data")

resume_checkpoint = None
resume_parts = sorted(Path("/content").glob("text2sql_resume_part_*"))
if resume_parts:
    if not RESUME_MANIFEST.exists():
        raise FileNotFoundError(RESUME_MANIFEST)
    expected_resume = json.loads(RESUME_MANIFEST.read_text(encoding="utf-8"))
    with RESUME_ARCHIVE.open("wb") as destination:
        for part in resume_parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if RESUME_ARCHIVE.stat().st_size != int(expected_resume["bytes"]):
        raise RuntimeError("Reassembled resume checkpoint size mismatch")
    if sha256(RESUME_ARCHIVE) != str(expected_resume["sha256"]):
        raise RuntimeError("Reassembled resume checkpoint checksum mismatch")
if RESUME_ARCHIVE.exists():
    output = ROOT / "output"
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RESUME_ARCHIVE, "r") as archive:
        output_root = output.resolve()
        members = archive.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            if target != output_root and output_root not in target.parents:
                raise RuntimeError(f"Unsafe resume member: {member.name}")
        archive.extractall(output, filter="data")
    checkpoint_dirs = sorted(
        (path for path in output.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if len(checkpoint_dirs) != 1:
        raise RuntimeError("Resume archive must contain exactly one checkpoint directory")
    resume_checkpoint = checkpoint_dirs[0].name

manifest = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = ROOT / item["path"]
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"Bundle verification failed: {item['path']}")
launch = json.loads((ROOT / "launch_config.json").read_text(encoding="utf-8"))
disk = shutil.disk_usage("/content")
versions = {
    name: importlib.metadata.version(name)
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes")
}
payload = {
    "gpu": gpu_name,
    "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "bf16": True,
    "disk_free_gib": round(disk.free / 1024**3, 2),
    "model": launch["model"]["slug"],
    "dataset_variant": launch["dataset_variant"],
    "smoke": launch["smoke"],
    "packages": versions,
    "resume_checkpoint": resume_checkpoint,
    "resume_parts": len(resume_parts),
}
(ROOT / "environment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("COLAB_QLORA_PREPARED=" + json.dumps(payload, sort_keys=True))
