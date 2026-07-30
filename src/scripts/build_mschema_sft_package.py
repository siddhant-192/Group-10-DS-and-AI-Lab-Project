#!/usr/bin/env python3
"""Render an existing Spider SFT package with the deployment M-Schema prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_xiyan_mschema_eval_data import PROJECT_ROOT, render_mschema


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "finetuning" / "spider_sft_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "finetuning" / "spider_mschema_sft_v1"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "spider"
FILES = ("train_base.jsonl", "train_curriculum.jsonl", "validation.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument(
        "--max-mschema-chars",
        type=int,
        default=10_000,
        help="Use the original bounded DDL prompt when a rendered M-Schema exceeds this size.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt(schema: str, question: str) -> str:
    return (
        "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
        f"【Schema】\n{schema}\n\n"
        f"【Question】\n{question}\n\n"
        "【Evidence】\n\n"
        "Please read and understand the database schema carefully, and generate an executable SQL based "
        "on the user's question and evidence. The generated SQL is protected by ```sql and ```."
    )


def render_file(
    source: Path,
    destination: Path,
    example_num: int,
    canonical: dict[str, dict[str, Any]],
    max_mschema_chars: int,
) -> dict[str, Any]:
    rows = read_jsonl(source)
    schema_cache: dict[tuple[str, str], str] = {}
    fallback_databases: set[str] = set()
    fallback_rows = 0
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            source_id = str(row.get("source_id", row["id"]))
            if source_id not in canonical:
                raise KeyError(f"Canonical Spider row not found for {source_id}")
            canonical_row = canonical[source_id]
            db_id = str(row["db_id"])
            if db_id != str(canonical_row["db_id"]):
                raise ValueError(f"Database mismatch for {source_id}")
            database_value = str(canonical_row["metadata"]["database_path"])
            database = PROJECT_ROOT / database_value
            key = (db_id, database_value)
            if key not in schema_cache:
                schema_cache[key] = render_mschema(database, db_id, example_num)
            schema = schema_cache[key]
            item = dict(row)
            item["question"] = str(canonical_row["question"])
            item["sql"] = str(canonical_row["sql"])
            use_ddl = len(schema) > max_mschema_chars
            if use_ddl:
                fallback_rows += 1
                fallback_databases.add(db_id)
                item["schema"] = str(canonical_row["schema"])
                item["messages"] = list(row["messages"])
                item["metadata"] = {
                    **dict(canonical_row["metadata"]),
                    "prompt_variant": "project_ddl_fallback_large_mschema",
                    "mschema_rendered_chars": len(schema),
                }
            else:
                item["schema"] = schema
                item["messages"] = [
                    {"role": "user", "content": prompt(schema, item["question"])},
                    {"role": "assistant", "content": item["sql"]},
                ]
                item["metadata"] = {
                    **dict(canonical_row["metadata"]),
                    "prompt_variant": "xiyan_official_mschema_english",
                    "mschema_example_num": example_num,
                }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return {
        "rows": len(rows),
        "databases": len(schema_cache),
        "ddl_fallback_rows": fallback_rows,
        "ddl_fallback_databases": sorted(fallback_databases),
    }


def main() -> int:
    args = parse_args()
    if args.examples < 0:
        raise ValueError("--examples cannot be negative")
    if args.max_mschema_chars < 1:
        raise ValueError("--max-mschema-chars must be positive")
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)
    processed_paths = (processed_dir / "train.jsonl", processed_dir / "validation.jsonl")
    for path in processed_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    canonical_rows = [row for path in processed_paths for row in read_jsonl(path)]
    canonical = {str(row["id"]): row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("Canonical Spider IDs are not unique")

    counts = {}
    for name in FILES:
        counts[name] = render_file(
            source_dir / name,
            output_dir / name,
            args.examples,
            canonical,
            args.max_mschema_chars,
        )
    checksums = {name: sha256(output_dir / name) for name in FILES}
    manifest = {
        "format_version": 1,
        "package": "spider_mschema_sft_v1",
        "source_package": str(source_dir),
        "prompt_variant": "xiyan_official_mschema_english",
        "mschema_example_num": args.examples,
        "max_mschema_chars": args.max_mschema_chars,
        "counts": counts,
        "source_checksums": {name: sha256(source_dir / name) for name in FILES},
    }
    (output_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "counts": counts, "checksums": checksums}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
