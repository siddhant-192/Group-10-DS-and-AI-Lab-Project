#!/usr/bin/env python3
"""Verify the exact selected Qwen3 QLoRA adapter release without loading it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "release" / "final_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def verify(adapter_dir: Path, manifest_path: Path) -> dict[str, Any]:
    adapter_dir = adapter_dir.resolve()
    manifest_path = manifest_path.resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest["base_model"]
    expected_adapter = manifest["adapter"]

    checked_files = 0
    for item in expected_adapter["expected_files"]:
        path = adapter_dir / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Required adapter file not found: {path}")
        require_equal(f"{item['path']} byte size", path.stat().st_size, item["bytes"])
        checked_files += 1

    weight_path = adapter_dir / expected_adapter["weight_file"]
    weight_hash = file_sha256(weight_path)
    require_equal("adapter weight SHA-256", weight_hash, expected_adapter["weight_sha256"])

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    require_equal("base model", config.get("base_model_name_or_path"), base["repo_id"])
    require_equal("base revision", config.get("revision"), base["revision"])

    expected_peft = expected_adapter["peft"]
    require_equal("PEFT type", config.get("peft_type"), expected_peft["type"])
    require_equal("LoRA rank", config.get("r"), expected_peft["r"])
    require_equal("LoRA alpha", config.get("lora_alpha"), expected_peft["lora_alpha"])
    require_equal("LoRA dropout", config.get("lora_dropout"), expected_peft["lora_dropout"])
    require_equal("LoRA bias", config.get("bias"), expected_peft["bias"])
    require_equal("task type", config.get("task_type"), expected_peft["task_type"])
    require_equal(
        "target modules",
        sorted(config.get("target_modules") or []),
        sorted(expected_peft["target_modules"]),
    )

    return {
        "release_id": manifest["release_id"],
        "adapter_dir": str(adapter_dir),
        "checked_files": checked_files,
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": weight_hash,
        "base_model": base["repo_id"],
        "base_revision": base["revision"],
        "status": "verified",
    }


def main() -> int:
    args = parse_args()
    result = verify(args.adapter_dir, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

