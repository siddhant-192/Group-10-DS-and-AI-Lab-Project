#!/usr/bin/env python3
"""Build the small code/data bundle uploaded to a Colab evaluation runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "text2sql_eval_models.json"
DEFAULT_MODEL_MANIFEST = PROJECT_ROOT / "models" / "text2sql-eval" / "download_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--candidate-selection",
        choices=("execution-consensus", "value-aware-voting"),
        default="execution-consensus",
    )
    parser.add_argument("--model-source", choices=("huggingface", "local"), default="huggingface")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--adapter-label")
    parser.add_argument(
        "--resume-predictions",
        type=Path,
        help="Seed one selected model's prediction file so evaluation resumes missing IDs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def add_file(archive: tarfile.TarFile, path: Path, arcname: str) -> dict[str, Any]:
    archive.add(path, arcname=arcname, recursive=False)
    return {"path": arcname, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    model_config = args.model_config.resolve()
    model_manifest = args.model_manifest.resolve()
    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir else None
    resume_predictions = args.resume_predictions.resolve() if args.resume_predictions else None
    if not data.exists():
        raise FileNotFoundError(data)
    if not model_manifest.exists():
        raise FileNotFoundError(
            f"Pinned model manifest is missing: {model_manifest}. Run scripts/download_eval_models.sh first."
        )
    if not model_config.exists():
        raise FileNotFoundError(f"Model configuration is missing: {model_config}")
    if args.model_source == "huggingface" and args.models:
        configured = {
            str(item["slug"]): item
            for item in json.loads(model_config.read_text(encoding="utf-8"))["models"]
        }
        pinned = {
            str(item["slug"]): item
            for item in json.loads(model_manifest.read_text(encoding="utf-8"))["models"]
        }
        for slug in args.models:
            merged = {**configured.get(str(slug), {}), **pinned.get(str(slug), {})}
            if not merged:
                raise ValueError(f"Selected model is not configured: {slug}")
            if not merged.get("revision"):
                raise ValueError(f"Hugging Face model is not pinned to an immutable revision: {slug}")
    if adapter_dir is not None:
        if not args.adapter_label:
            raise ValueError("--adapter-label is required with --adapter-dir")
        if not adapter_dir.is_dir():
            raise FileNotFoundError(adapter_dir)
        if not args.models or len(args.models) != 1:
            raise ValueError("Adapter evaluation requires exactly one --model")
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise FileNotFoundError(adapter_dir / "adapter_model.safetensors")
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(adapter_dir / "adapter_config.json")
    elif args.adapter_label:
        raise ValueError("--adapter-label requires --adapter-dir")

    rows = load_rows(data, args.limit)
    if not rows:
        raise ValueError("No validation examples selected")
    resume_slug = None
    resumed_count = 0
    if resume_predictions is not None:
        if not resume_predictions.is_file():
            raise FileNotFoundError(resume_predictions)
        if not args.models or len(args.models) != 1:
            raise ValueError("Prediction resume requires exactly one --model")
        resume_slug = str(args.models[0])
        completed = load_rows(resume_predictions, None)
        completed_ids = [str(row.get("id", "")) for row in completed]
        selected_ids = {str(row["id"]) for row in rows}
        if not completed_ids or len(completed_ids) != len(set(completed_ids)):
            raise ValueError("Resume predictions must have non-empty, unique IDs")
        unknown = set(completed_ids) - selected_ids
        if unknown:
            raise ValueError(f"Resume predictions contain {len(unknown)} IDs outside selected data")
        resumed_count = len(completed_ids)
    database_paths = sorted(
        {
            (PROJECT_ROOT / str(row["metadata"]["database_path"])).resolve()
            for row in rows
        }
    )
    missing = [path for path in database_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing validation databases: {missing[:3]}")

    launch_config = {
        "models": args.models,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "candidate_selection": args.candidate_selection,
        "model_source": args.model_source,
        "adapter_dir": "/content/text2sql_eval/adapter" if adapter_dir else None,
        "adapter_label": args.adapter_label,
    }
    temporary_launch = output.parent / "launch_config.json"
    temporary_launch.write_text(json.dumps(launch_config, indent=2) + "\n", encoding="utf-8")
    bundle_manifest: dict[str, Any] = {
        "format_version": 1,
        "validation_examples": len(rows),
        "validation_databases": len(database_paths),
        "adapter_label": args.adapter_label,
        "files": [],
    }
    temporary_manifest = output.parent / "bundle_manifest.json"

    required = [
        (PROJECT_ROOT / "scripts" / "evaluate_text2sql_models.py", "scripts/evaluate_text2sql_models.py"),
        (PROJECT_ROOT / "scripts" / "evaluate_finer_vllm.py", "scripts/evaluate_finer_vllm.py"),
        (model_config, "configs/text2sql_eval_models.json"),
        (model_manifest, "models/text2sql-eval/download_manifest.json"),
        (data, "data/processed/spider/validation.jsonl"),
        (temporary_launch, "launch_config.json"),
    ]
    try:
        with tarfile.open(output, "w:gz") as archive:
            for path, arcname in required:
                bundle_manifest["files"].append(add_file(archive, path, arcname))
            for path in database_paths:
                arcname = path.relative_to(PROJECT_ROOT).as_posix()
                bundle_manifest["files"].append(add_file(archive, path, arcname))
            if adapter_dir is not None:
                for path in sorted(adapter_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    arcname = (Path("adapter") / path.relative_to(adapter_dir)).as_posix()
                    bundle_manifest["files"].append(add_file(archive, path, arcname))
            if resume_predictions is not None and resume_slug is not None:
                bundle_manifest["files"].append(
                    add_file(
                        archive,
                        resume_predictions,
                        f"results/{resume_slug}/predictions.jsonl",
                    )
                )
            temporary_manifest.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")
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
                "validation_examples": len(rows),
                "validation_databases": len(database_paths),
                "adapter_label": args.adapter_label,
                "resumed_predictions": resumed_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
