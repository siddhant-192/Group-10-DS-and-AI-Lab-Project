#!/usr/bin/env python3
"""Render Spider validation with XiYan's official M-Schema-style prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "spider" / "validation_xiyan_mschema.jsonl"
EMAIL = re.compile(r"^[\w.-]+@[\w.-]+\.\w+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def examples_for(connection: sqlite3.Connection, table: str, column: str, limit: int = 5) -> list[str]:
    try:
        rows = connection.execute(
            f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column)} IS NOT NULL LIMIT ?", (limit,)
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    values = [str(row[0]) for row in rows if row[0] is not None and str(row[0])]
    if any(EMAIL.match(value) or "http://" in value or "https://" in value for value in values):
        return []
    return values


def render_mschema(database: Path, db_id: str, example_num: int) -> str:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        tables = [
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        output = [f"【DB_ID】 {db_id}", "【Schema】"]
        foreign_keys = []
        for table in tables:
            output.append(f"# Table: {table}")
            field_lines = []
            for column in connection.execute(f"PRAGMA table_info({quote_identifier(table)})"):
                _cid, name, raw_type, _not_null, _default, primary_key = column
                field_type = str(raw_type or "").split("(", 1)[0].upper()
                line = f"({name}:{field_type}"
                if primary_key:
                    line += ", Primary Key"
                values = examples_for(connection, table, str(name))
                values = [value for value in values if value is not None]
                if values and example_num > 0:
                    if field_type in {"DATE", "TIME", "DATETIME", "TIMESTAMP"}:
                        values = values[:1]
                    elif max(map(len, values)) > 50:
                        values = []
                    elif max(map(len, values)) > 20:
                        values = values[:1]
                    else:
                        values = values[:example_num]
                    if values:
                        line += f", Examples: [{', '.join(values)}]"
                line += ")"
                field_lines.append(line)
            output.extend(("[", ",\n".join(field_lines), "]"))
            for foreign in connection.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})"):
                _id, _seq, referred_table, constrained, referred, *_rest = foreign
                foreign_keys.append(f"{table}.{constrained}={referred_table}.{referred}")
        if foreign_keys:
            output.append("【Foreign keys】")
            output.extend(foreign_keys)
        return "\n".join(output)
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    if args.examples < 0:
        raise ValueError("--examples cannot be negative")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rendered = []
    cache: dict[str, str] = {}
    for row in rows:
        db_id = str(row["db_id"])
        database = PROJECT_ROOT / str(row["metadata"]["database_path"])
        if db_id not in cache:
            cache[db_id] = render_mschema(database, db_id, args.examples)
        question = str(row["question"])
        prompt = (
            "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
            f"【Schema】\n{cache[db_id]}\n\n"
            f"【Question】\n{question}\n\n"
            "【Evidence】\n\n"
            "Please read and understand the database schema carefully, and generate an executable SQL based "
            "on the user's question and evidence. The generated SQL is protected by ```sql and ```."
        )
        item = dict(row)
        item["schema"] = cache[db_id]
        item["messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(row["sql"])},
        ]
        item["metadata"] = {
            **dict(row["metadata"]),
            "prompt_variant": "xiyan_official_mschema_english",
            "mschema_example_num": args.examples,
        }
        rendered.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rendered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({"output": str(output_path), "examples": len(rendered), "databases": len(cache)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
