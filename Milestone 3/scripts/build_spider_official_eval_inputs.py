#!/usr/bin/env python3
"""Build Spider's gold, prediction, and foreign-key files from aligned project artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--schemas", type=Path, default=PROJECT_ROOT / "data/processed/spider/schemas.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_line(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.replace("\r", " ").replace("\n", " ")).strip() or "SELECT 1"


def spider_table_entry(database: dict) -> dict:
    tables = database["tables"]
    table_names = [str(table["name"]) for table in tables]
    table_index = {name.lower(): index for index, name in enumerate(table_names)}
    columns: list[list[int | str]] = [[-1, "*"]]
    column_index: dict[tuple[str, str], int] = {}
    column_types = ["text"]
    primary_keys: list[int] = []
    for index, table in enumerate(tables):
        for column in table["columns"]:
            column_id = len(columns)
            name = str(column["name"])
            columns.append([index, name])
            column_index[(table_names[index].lower(), name.lower())] = column_id
            declared = str(column.get("declared_type") or "text").lower()
            column_types.append(
                "number" if any(token in declared for token in ("int", "real", "float", "double", "decimal", "numeric"))
                else "time" if any(token in declared for token in ("date", "time", "year"))
                else "boolean" if "bool" in declared
                else "text"
            )
            if int(column.get("primary_key_position") or 0) > 0:
                primary_keys.append(column_id)
    foreign_keys: list[list[int]] = []
    for table in tables:
        source_table = str(table["name"]).lower()
        for foreign_key in table["foreign_keys"]:
            target_table = str(foreign_key["referenced_table"]).lower()
            source = column_index.get((source_table, str(foreign_key["from_column"]).lower()))
            target = column_index.get((target_table, str(foreign_key["referenced_column"]).lower()))
            if source is not None and target is not None:
                foreign_keys.append([source, target])
    return {
        "db_id": database["db_id"],
        "table_names_original": table_names,
        "table_names": [name.lower().replace("_", " ") for name in table_names],
        "column_names_original": columns,
        "column_names": [[table, str(name).lower().replace("_", " ")] for table, name in columns],
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }


def main() -> int:
    args = parse_args()
    validation = read_jsonl(args.validation.resolve())
    if args.limit is not None:
        validation = validation[: args.limit]
    predictions = {str(row["id"]): row for row in read_jsonl(args.predictions.resolve())}
    ids = [str(row["id"]) for row in validation]
    if not set(ids).issubset(predictions):
        raise ValueError("Predictions do not cover all selected validation rows")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "gold.sql").write_text(
        "".join(f"{one_line(str(row['sql']))}\t{row['db_id']}\n" for row in validation),
        encoding="utf-8",
    )
    (output / "pred.sql").write_text(
        "".join(f"{one_line(str(predictions[item].get('predicted_sql') or ''))}\n" for item in ids),
        encoding="utf-8",
    )
    schema_payload = json.loads(args.schemas.resolve().read_text(encoding="utf-8"))
    wanted = {str(row["db_id"]) for row in validation}
    tables = [
        spider_table_entry(database)
        for db_id, database in schema_payload["databases"].items()
        if db_id in wanted
    ]
    (output / "tables.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "examples": len(validation),
        "databases": len(wanted),
        "validation": str(args.validation.resolve()),
        "predictions": str(args.predictions.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
