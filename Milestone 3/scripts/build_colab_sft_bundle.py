#!/usr/bin/env python3
"""Build the code/data/config bundle for one Colab QLoRA training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "text2sql_qlora_training.json"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "text2sql_eval_models.json"
DEFAULT_MODEL_MANIFEST = PROJECT_ROOT / "models" / "text2sql-eval" / "download_manifest.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "finetuning" / "spider_sft_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-4b-instruct-2507")
    parser.add_argument("--dataset-variant", choices=("base", "curriculum"), default="curriculum")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--no-resume-smoke-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, path: Path, arcname: str) -> dict[str, Any]:
    archive.add(path, arcname=arcname, recursive=False)
    return {"path": arcname, "bytes": path.stat().st_size, "sha256": sha256(path)}


def find_model(slug: str, config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    configs = {str(item["slug"]): item for item in config["models"]}
    manifests = {str(item["slug"]): item for item in manifest["models"]}
    if slug not in configs or slug not in manifests:
        raise ValueError(f"Model is not configured and pinned: {slug}")
    return {**configs[slug], **manifests[slug]}


def merge_training(
    full_config: dict[str, Any], model: str, smoke: bool, args: argparse.Namespace
) -> dict[str, Any]:
    mode = dict(full_config["smoke" if smoke else "full"])
    model_values = dict(full_config["models"][model])
    optimization = dict(full_config["optimization"])
    training = {**optimization, **model_values, **mode}
    # Smoke settings deliberately override model batch sizes. Explicit CLI values
    # override every file-based setting.
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.train_limit is not None:
        training["train_limit"] = args.train_limit
    if args.validation_limit is not None:
        training["validation_limit"] = args.validation_limit
    if int(training["max_steps"]) == 0:
        raise ValueError("max_steps cannot be zero")
    if smoke and int(training["max_steps"]) <= int(mode["resume_phase_one_steps"]):
        raise ValueError("Smoke max_steps must exceed resume_phase_one_steps")
    return training


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    training_config_path = args.training_config.resolve()
    model_config_path = args.model_config.resolve()
    model_manifest_path = args.model_manifest.resolve()
    data_dir = args.data_dir.resolve()
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model = find_model(args.model, model_config, model_manifest)
    training = merge_training(training_config, args.model, args.smoke, args)
    train_name = "train_base.jsonl" if args.dataset_variant == "base" else "train_curriculum.jsonl"
    train_path = data_dir / train_name
    validation_path = data_dir / "validation.jsonl"
    data_manifest = data_dir / "manifest.json"
    data_checksums = data_dir / "checksums.json"
    for path in (
        train_path,
        validation_path,
        data_manifest,
        data_checksums,
        training_config_path,
        model_config_path,
        model_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    expected = json.loads(data_checksums.read_text(encoding="utf-8"))
    for path in (train_path, validation_path):
        if expected.get(path.name) != sha256(path):
            raise RuntimeError(f"SFT package checksum mismatch: {path}")

    launch_config = {
        "run_name": args.run_name,
        "smoke": args.smoke,
        "resume_smoke_test": bool(args.smoke and not args.no_resume_smoke_test),
        "model": {
            key: model[key]
            for key in (
                "slug",
                "tier",
                "repo_id",
                "revision",
                "trust_remote_code",
                "base_parameter_count",
            )
        },
        "dataset_variant": args.dataset_variant,
        "train_data": "/content/text2sql_sft/data/train.jsonl",
        "validation_data": "/content/text2sql_sft/data/validation.jsonl",
        "training": training,
        "quantization": training_config["quantization"],
        "lora": training_config["lora"],
        "resume_phase_one_steps": (
            int(training_config["smoke"]["resume_phase_one_steps"]) if args.smoke else None
        ),
    }
    temporary_launch = output.parent / "sft_launch_config.json"
    temporary_manifest = output.parent / "sft_bundle_manifest.json"
    temporary_launch.write_text(json.dumps(launch_config, indent=2) + "\n", encoding="utf-8")
    bundle_manifest: dict[str, Any] = {
        "format_version": 1,
        "run_name": args.run_name,
        "model": model["slug"],
        "dataset_variant": args.dataset_variant,
        "smoke": args.smoke,
        "files": [],
    }
    files = (
        (PROJECT_ROOT / "scripts" / "train_text2sql_qlora.py", "scripts/train_text2sql_qlora.py"),
        (training_config_path, "configs/text2sql_qlora_training.json"),
        (model_config_path, "configs/text2sql_eval_models.json"),
        (model_manifest_path, "models/text2sql-eval/download_manifest.json"),
        (train_path, "data/train.jsonl"),
        (validation_path, "data/validation.jsonl"),
        (data_manifest, "data/source_manifest.json"),
        (data_checksums, "data/source_checksums.json"),
        (temporary_launch, "launch_config.json"),
    )
    try:
        with tarfile.open(output, "w:gz") as archive:
            for path, arcname in files:
                bundle_manifest["files"].append(add_file(archive, path, arcname))
            temporary_manifest.write_text(
                json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8"
            )
            archive.add(temporary_manifest, arcname="bundle_manifest.json", recursive=False)
    finally:
        temporary_launch.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "bundle": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "run_name": args.run_name,
                "model": model["slug"],
                "dataset_variant": args.dataset_variant,
                "smoke": args.smoke,
                "train_limit": training.get("train_limit"),
                "validation_limit": training.get("validation_limit"),
                "max_steps": training["max_steps"],
                "resume_smoke_test": launch_config["resume_smoke_test"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
