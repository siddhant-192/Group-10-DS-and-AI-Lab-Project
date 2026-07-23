#!/usr/bin/env python3
"""Validate local chat templates and token lengths before allocating Colab."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "evaluate_text2sql_models.py"
SPEC = importlib.util.spec_from_file_location("evaluate_text2sql_models", EVALUATOR_PATH)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main() -> int:
    args = parse_args()
    config = json.loads((PROJECT_ROOT / "configs" / "text2sql_eval_models.json").read_text())["models"]
    rows = evaluator.read_jsonl(PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl")
    report = []
    for spec in config:
        model_dir = PROJECT_ROOT / "models" / "text2sql-eval" / spec["slug"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
        )
        lengths = []
        target_lengths = []
        for row in rows:
            prompt = evaluator.render_prompt(tokenizer, row)
            lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
            target_lengths.append(len(tokenizer(str(row["sql"]), add_special_tokens=False)["input_ids"]))
        item = {
            "slug": spec["slug"],
            "examples": len(lengths),
            "min": min(lengths),
            "median": int(statistics.median(lengths)),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "over_limit": sum(value > args.max_input_tokens for value in lengths),
            "target_p95": percentile(target_lengths, 0.95),
            "target_max": max(target_lengths),
            "targets_over_generation_limit": sum(value > args.max_new_tokens for value in target_lengths),
        }
        report.append(item)
        print(json.dumps(item, sort_keys=True))
    if any(item["over_limit"] or item["targets_over_generation_limit"] for item in report):
        raise RuntimeError(
            f"One or more prompts exceed {args.max_input_tokens} input tokens or "
            f"{args.max_new_tokens} target tokens"
        )
    output = PROJECT_ROOT / "models" / "text2sql-eval" / "input_preflight.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"INPUT_PREFLIGHT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
