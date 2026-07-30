#!/usr/bin/env python3
"""Render Spider data in FINER-SQL's DDL or official enriched prompt format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "spider" / "validation_finer_official.jsonl"
DEFAULT_OFFICIAL_PROMPTS = PROJECT_ROOT / "data" / "raw" / "finer-sql" / "spider_dev_prompts.parquet"
SYSTEM = """You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.
Rules:
- Output exactly one SQL statement.
- The SQL must be executable on SQLite.
- Do not include any explanatory text.
- Output one SQL statement only. Do not include any extra text, tags, or code fences."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--schema-source",
        choices=("official", "ddl"),
        default="official",
        help="Use FINER's published value-enriched prompts or the project's plain DDL.",
    )
    parser.add_argument("--official-prompts-parquet", type=Path, default=DEFAULT_OFFICIAL_PROMPTS)
    return parser.parse_args()


def load_official_prompts(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Official prompt rendering requires pyarrow; use .venv-data/bin/python."
        ) from exc
    if not path.exists():
        raise FileNotFoundError(path)
    return parquet.read_table(path).to_pylist()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    official_rows = (
        load_official_prompts(args.official_prompts_parquet.resolve())
        if args.schema_source == "official"
        else None
    )
    if official_rows is not None and len(official_rows) != len(rows):
        raise ValueError(
            f"Official prompt/local validation length mismatch: {len(official_rows)} != {len(rows)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            item = dict(row)
            if official_rows is not None:
                official = official_rows[index]
                if int(official["sample_id"]) != index:
                    raise ValueError(f"Unexpected official sample_id at row {index}: {official['sample_id']}")
                if str(official["db_id"]) != str(row["db_id"]):
                    raise ValueError(f"Database mismatch at row {index}")
                if str(official["question"]).strip() != str(row["question"]).strip():
                    raise ValueError(f"Question mismatch at row {index}")
                item["messages"] = [
                    {"role": str(message["role"]), "content": str(message["content"])}
                    for message in official["messages"]
                ] + [{"role": "assistant", "content": str(row["sql"])}]
                prompt_variant = "finer_sql_official_value_enriched"
            else:
                item["messages"] = [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Database Schema:\n{row['schema']}\n\n Question: {row['question']}",
                    },
                    {"role": "assistant", "content": str(row["sql"])},
                ]
                prompt_variant = "finer_sql_published_instruction_ddl"
            item["metadata"] = {
                **dict(row["metadata"]),
                "prompt_variant": prompt_variant,
            }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {"output": str(output_path), "examples": len(rows), "schema_source": args.schema_source},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
