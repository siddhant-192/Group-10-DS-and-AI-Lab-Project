from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_final_adapter.py"
SPEC = importlib.util.spec_from_file_location("verify_final_adapter", SCRIPT)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def make_release(tmp_path: Path) -> tuple[Path, Path]:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    weight = adapter_dir / "adapter_model.safetensors"
    weight.write_bytes(b"test-adapter")
    config = {
        "base_model_name_or_path": "org/base",
        "revision": "abc123",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj"],
    }
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "release_id": "test-release",
        "base_model": {"repo_id": "org/base", "revision": "abc123"},
        "adapter": {
            "weight_file": weight.name,
            "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "expected_files": [
                {"path": weight.name, "bytes": weight.stat().st_size},
                {"path": config_path.name, "bytes": config_path.stat().st_size},
            ],
            "peft": {
                "type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "v_proj"],
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return adapter_dir, manifest_path


def test_verify_accepts_exact_adapter(tmp_path: Path) -> None:
    adapter_dir, manifest_path = make_release(tmp_path)
    result = verifier.verify(adapter_dir, manifest_path)
    assert result["status"] == "verified"
    assert result["release_id"] == "test-release"
    assert result["checked_files"] == 2


def test_verify_rejects_changed_weights(tmp_path: Path) -> None:
    adapter_dir, manifest_path = make_release(tmp_path)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"changed-value")
    with pytest.raises(ValueError, match="byte size|SHA-256"):
        verifier.verify(adapter_dir, manifest_path)
