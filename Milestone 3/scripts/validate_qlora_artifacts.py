#!/usr/bin/env python3
"""Validate a downloaded QLoRA adapter and its newest resumable checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    downloaded = run_dir / "downloaded"
    output = downloaded / "output"
    status = read_json(downloaded / "status.json")
    launch = read_json(downloaded / "launch_config.json")
    run_manifest = read_json(output / "run_manifest.json")
    phases = read_jsonl(output / "phase_history.jsonl")
    tokenization = read_json(output / "tokenization_summary.json")
    checkpoints = sorted(output.glob("checkpoint-*"), key=checkpoint_step)
    if status.get("phase") != "complete":
        raise RuntimeError(f"Run is not complete: {status}")
    if not checkpoints:
        raise FileNotFoundError("No resumable checkpoint was downloaded")
    latest = checkpoints[-1]
    trainer_state = read_json(latest / "trainer_state.json")
    adapter = output / "final_adapter" / "adapter_model.safetensors"
    checkpoint_adapter = latest / "adapter_model.safetensors"
    adapter_config = read_json(output / "final_adapter" / "adapter_config.json")
    required_checkpoint_files = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    )
    missing = [name for name in required_checkpoint_files if not (latest / name).exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is not resumable; missing: {missing}")
    if sha256(adapter) != sha256(checkpoint_adapter):
        raise RuntimeError("Final adapter weights do not match the newest checkpoint")
    if int(trainer_state["global_step"]) != int(status["step"]):
        raise RuntimeError("Trainer state and final status disagree on global step")
    if tokenization.get("truncated_examples") != 0:
        raise RuntimeError("Training reported truncated examples")
    if max(tokenization["train_max_tokens"], tokenization["validation_max_tokens"]) > int(
        tokenization["max_seq_length"]
    ):
        raise RuntimeError("Tokenization summary exceeds max_seq_length")
    revision = str(launch["model"]["revision"])
    if adapter_config.get("revision") != revision:
        raise RuntimeError(
            f"Adapter revision is not pinned: expected={revision}, found={adapter_config.get('revision')}"
        )
    resume_verified = any(phase.get("resumed_from_checkpoint") for phase in phases)
    if launch.get("resume_smoke_test") and not resume_verified:
        raise RuntimeError("Smoke run did not prove checkpoint resume")
    model_config = read_json(PROJECT_ROOT / "configs" / "text2sql_eval_models.json")
    configured_model = next(
        item for item in model_config["models"] if item["slug"] == launch["model"]["slug"]
    )
    logical_base_parameters = int(configured_model["base_parameter_count"])
    trainable_parameters = int(run_manifest["trainable_parameters"])
    peak_allocated = max(float(phase["cuda_peak"]["max_allocated_gib"]) for phase in phases)
    peak_reserved = max(float(phase["cuda_peak"]["max_reserved_gib"]) for phase in phases)
    report = {
        "validated": True,
        "run_dir": str(run_dir),
        "model": launch["model"],
        "dataset_variant": launch["dataset_variant"],
        "smoke": launch["smoke"],
        "global_step": int(status["step"]),
        "checkpoint": latest.name,
        "checkpoint_resume_files_complete": True,
        "resume_verified": resume_verified,
        "phase_count": len(phases),
        "explicit_assistant_only_labels": bool(run_manifest["explicit_assistant_only_labels"]),
        "truncated_examples": tokenization["truncated_examples"],
        "train_min_tokens": tokenization["train_min_tokens"],
        "train_max_tokens": tokenization["train_max_tokens"],
        "validation_min_tokens": tokenization["validation_min_tokens"],
        "validation_max_tokens": tokenization["validation_max_tokens"],
        "final_eval_loss": float(status["eval_loss"]),
        "trainable_parameters": trainable_parameters,
        "logical_base_parameters": logical_base_parameters,
        "trainable_parameter_pct_of_logical_base": round(
            100.0 * trainable_parameters / logical_base_parameters, 6
        ),
        "overall_peak_allocated_gib": peak_allocated,
        "overall_peak_reserved_gib": peak_reserved,
        "adapter_bytes": adapter.stat().st_size,
        "adapter_sha256": sha256(adapter),
        "checkpoint_adapter_matches_final": True,
        "base_revision_pinned_in_adapter": True,
        "raw_run_manifest_note": (
            "For runs created before logical parameter accounting was added, run_manifest.total_parameters "
            "counts packed 4-bit storage elements. This validation report uses the exact unquantized tensor count."
        ),
    }
    destination = run_dir / "artifact-validation.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"QLORA_ARTIFACT_VALIDATION={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
