#!/usr/bin/env python3
"""Aggregate compact Qwen3 HPO training and generation-evaluation artifacts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "artifacts" / "qlora-hparam" / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "qlora-hparam"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def latest_evaluation(label: str) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        PROJECT_ROOT.glob(f"artifacts/zero-shot-eval/runs/*/downloaded/results/q3hp-{label}/metrics.json")
    )
    if not candidates:
        return None, None
    path = candidates[-1]
    return path, read_json(path)


def compatible_evaluation(label: str) -> tuple[Path | None, dict[str, Any] | None]:
    path = OUTPUT_ROOT / "macsql" / label / "metrics.json"
    if not path.is_file():
        return None, None
    metrics = read_json(path)
    # The standalone scorer writes a label-keyed aggregate when invoked with
    # LABEL=predictions.jsonl; accept both that form and the flat legacy form.
    if "macsql_execution" not in metrics and len(metrics) == 1:
        nested = next(iter(metrics.values()))
        if isinstance(nested, dict):
            metrics = nested
    return path, metrics


def collect() -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(RUN_ROOT.glob("*")):
        label_path = run_dir / "trial-label.txt"
        config_path = run_dir / "training-config.json"
        manifest_path = run_dir / "downloaded" / "output" / "run_manifest.json"
        if not label_path.is_file() or not config_path.is_file() or not manifest_path.is_file():
            continue
        label = label_path.read_text(encoding="utf-8").strip()
        config = read_json(config_path)
        manifest = read_json(manifest_path)
        last = manifest.get("last_phase", {})
        eval_path, evaluation = latest_evaluation(label)
        compatible_path, compatible = compatible_evaluation(label)
        row = {
            "label": label,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
            "learning_rate": config["optimization"]["learning_rate"],
            "rank": config["lora"]["r"],
            "alpha": config["lora"]["lora_alpha"],
            "dropout": config["lora"]["lora_dropout"],
            "target_modules": config["lora"]["target_modules"],
            "epochs": config["optimization"]["num_train_epochs"],
            "scheduler": config["optimization"]["lr_scheduler_type"],
            "warmup_ratio": config["optimization"]["warmup_ratio"],
            "weight_decay": config["optimization"]["weight_decay"],
            "seed": config["optimization"]["seed"],
            "train_batch_size": config["models"]["qwen3-4b-instruct-2507"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["models"]["qwen3-4b-instruct-2507"]["gradient_accumulation_steps"],
            "train_examples": manifest.get("train_data", {}).get("selected_examples"),
            "validation_examples": manifest.get("validation_data", {}).get("selected_examples"),
            "global_step": last.get("global_step"),
            "train_loss": last.get("train_metrics", {}).get("train_loss"),
            "eval_loss": last.get("eval_metrics", {}).get("eval_loss"),
            "train_runtime_seconds": last.get("train_metrics", {}).get("train_runtime"),
            "peak_allocated_gib": last.get("cuda_peak", {}).get("max_allocated_gib"),
            "adapter_bytes": last.get("adapter_bytes"),
            "adapter_dir": str((run_dir / "downloaded" / "output" / "final_adapter").relative_to(PROJECT_ROOT)),
            "evaluation_metrics": str(eval_path.relative_to(PROJECT_ROOT)) if eval_path else None,
            "strict_execution_pct": evaluation.get("execution_match_pct") if evaluation else None,
            "compatible_execution_pct": (
                compatible.get("macsql_execution", {}).get("pct") if compatible else None
            ),
            "compatible_metrics": str(compatible_path.relative_to(PROJECT_ROOT)) if compatible_path else None,
            "syntax_valid_pct": evaluation.get("syntax_valid_pct") if evaluation else None,
            "normalized_exact_match_pct": evaluation.get("normalized_exact_match_pct") if evaluation else None,
            "mean_generation_ms": evaluation.get("mean_generation_ms_per_example") if evaluation else None,
        }
        selected[label] = row
    return [selected[label] for label in sorted(selected)]


def main() -> int:
    rows = collect()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_text(OUTPUT_ROOT / "search_summary.json", json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n")
    columns = list(rows[0]) if rows else ["label"]
    csv_path = OUTPUT_ROOT / "search_summary.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    lines = [
        "# Qwen3 QLoRA hyperparameter search",
        "",
        "Selection uses database-disjoint generated-SQL strict execution accuracy; loss is diagnostic.",
        "",
        "| Trial | LR | r/alpha | Dropout | Train rows | Eval loss | Strict EX | Compatible EX | Syntax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = lambda key: "pending" if row.get(key) is None else str(round(float(row[key]), 6))
        lines.append(
            f"| {row['label']} | {row['learning_rate']} | {row['rank']}/{row['alpha']} | "
            f"{row['dropout']} | {row['train_examples']} | {value('eval_loss')} | "
            f"{value('strict_execution_pct')} | {value('compatible_execution_pct')} | "
            f"{value('syntax_valid_pct')} |"
        )
    atomic_text(OUTPUT_ROOT / "search_summary.md", "\n".join(lines) + "\n")
    print(json.dumps({"trials": len(rows), "output": str(OUTPUT_ROOT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
