#!/usr/bin/env python3

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare BIRD Mini-Dev SQLite evaluation data."
    )

    parser.add_argument(
        "--bird-root",
        type=Path,
        required=True,
        help=(
            "Path to the BIRD Mini-Dev root containing "
            "mini_dev_sqlite.json, mini_dev_sqlite_gold.sql, "
            "and dev_databases/."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evidence/milestone5/bird"),
        help="Directory for the prepared evaluation JSONL.",
    )

    parser.add_argument(
        "--gold-timeout",
        type=float,
        default=15.0,
        help="Seconds to allow each gold query to run during validation.",
    )

    return parser.parse_args()


def find_database(database_dir, db_id):
    db_dir = database_dir / db_id

    candidates = [
        db_dir / f"{db_id}.sqlite",
        db_dir / f"{db_id}.db",
    ]

    for path in candidates:
        if path.exists():
            return path

    for pattern in ("*.sqlite", "*.db"):
        matches = sorted(db_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No SQLite database found for db_id={db_id} "
        f"in {db_dir}"
    )


def get_schema(db_path):
    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
    )

    try:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        schema_parts = []

        for (table_name,) in tables:
            columns = connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

            column_names = [row[1] for row in columns]

            schema_parts.append(
                f"CREATE TABLE {table_name} "
                f"({', '.join(column_names)});"
            )

        return "\n".join(schema_parts)

    finally:
        connection.close()


def validate_gold_sql(db_path, sql, timeout):
    """
    Executes the gold query read-only, purely to RECORD whether it runs --
    does NOT exclude or modify anything. Mirrors what the eval script itself
    does at scoring time (gold.status == "ok"), just surfaced earlier so you
    have visibility before spending GPU time on the full eval.
    """
    t0 = time.monotonic()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.execute(sql).fetchmany(1)
            return {"status": "ok", "error": None, "elapsed": time.monotonic() - t0}
        finally:
            conn.close()
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "elapsed": time.monotonic() - t0,
        }


def resolve_gold_sql(raw_line, db_id, db_path, timeout):
    """
    Determine the correct gold SQL text from a raw gold-file line that may
    or may not have a trailing db_id appended.

    Whitespace alone can't reliably tell corruption apart from a query
    that legitimately ends in a table name matching the db_id (e.g.
    "...FROM superhero" in the superhero database) -- both patterns show
    up with a SINGLE space in this file, so counting spaces is not a safe
    signal on its own.

    Instead: try the line exactly as written first. If it's already valid
    SQL, keep it untouched (this correctly preserves legitimate endings
    like "FROM superhero"). Only if it's NOT valid do we try stripping a
    trailing db_id (regardless of what whitespace precedes it -- tab, one
    space, or several) and check whether THAT becomes valid.
    """
    full_candidate = raw_line.strip()
    full_check = validate_gold_sql(db_path, full_candidate, timeout)

    if full_check["status"] == "ok":
        return full_candidate, full_check

    stripped_candidate = re.sub(
        rf"\s+{re.escape(db_id)}\s*$",
        "",
        raw_line,
    ).strip()

    if stripped_candidate != full_candidate:
        stripped_check = validate_gold_sql(db_path, stripped_candidate, timeout)
        if stripped_check["status"] == "ok":
            return stripped_candidate, stripped_check
        # Neither the raw line nor the stripped version executes --
        # prefer the stripped version (it's the best guess at intended
        # SQL) but this row will correctly show up as gold_execution_status
        # = error either way, which is the honest outcome here.
        return stripped_candidate, stripped_check

    return full_candidate, full_check


def main():
    args = parse_args()

    bird_root = args.bird_root.resolve()
    output_dir = args.output_dir.resolve()

    questions_file = bird_root / "mini_dev_sqlite.json"
    gold_file = bird_root / "mini_dev_sqlite_gold.sql"
    database_dir = bird_root / "dev_databases"

    # Validate input layout.
    for path in (questions_file, gold_file, database_dir):
        if not path.exists():
            raise FileNotFoundError(
                f"Required BIRD file/directory not found: {path}"
            )

    print(f"BIRD root: {bird_root}")
    print(f"Output directory: {output_dir}")

    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)

    with open(gold_file, encoding="utf-8") as f:
        gold_lines = [
            line.rstrip("\n")
            for line in f
        ]

    if len(questions) != 500:
        raise RuntimeError(
            f"Expected 500 BIRD questions, got {len(questions)}"
        )

    if len(gold_lines) != 500:
        raise RuntimeError(
            f"Expected 500 gold SQL lines, got {len(gold_lines)}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "bird_eval.jsonl"
    validation_summary_file = output_dir / "gold_validation.json"

    gold_status_counts = {"ok": 0, "error": 0}
    gold_errors_sample = []

    with open(output_file, "w", encoding="utf-8") as out:

        for index, (question, gold_line) in enumerate(
            zip(questions, gold_lines)
        ):
            db_id = question["db_id"]

            db_path = find_database(
                database_dir,
                db_id,
            )

            schema = get_schema(db_path)

            # Resolve the gold SQL using actual validity, not whitespace
            # guessing -- see resolve_gold_sql() docstring for why.
            gold_sql, gold_check = resolve_gold_sql(
                gold_line, db_id, db_path, args.gold_timeout
            )

            gold_status_counts[gold_check["status"]] = (
                gold_status_counts.get(gold_check["status"], 0) + 1
            )
            if gold_check["status"] != "ok" and len(gold_errors_sample) < 20:
                gold_errors_sample.append({
                    "id": index,
                    "db_id": db_id,
                    "error": gold_check["error"],
                })

            row = {
                "id": index,
                "db_id": db_id,
                "question": question["question"],
                "sql": gold_sql,
                "schema": schema,
                # NOTE: nested under "metadata" -- evaluate_text2sql_models.py's
                # resolve_database() reads row["metadata"]["database_path"].
                # Putting this at the top level (as before) silently resolves
                # to the project root for every row and breaks execution for
                # ALL examples without raising an error.
                "metadata": {
                    "database_path": str(db_path),
                    "gold_execution_status_at_prep": gold_check["status"],
                    "gold_execution_error_at_prep": gold_check["error"],
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a text-to-SQL system. "
                            "Generate only the SQL query."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Database dialect: SQLite\n\n"
                            f"Database schema:\n{schema}\n\n"
                            f"Question: {question['question']}"
                        ),
                    },
                ],
            }

            out.write(
                json.dumps(row) + "\n"
            )

            if (index + 1) % 50 == 0:
                print(
                    f"Prepared {index + 1}/500  "
                    f"(gold ok={gold_status_counts.get('ok', 0)}, "
                    f"error={gold_status_counts.get('error', 0)})"
                )

    with open(validation_summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_examples": 500,
                "gold_status_counts": gold_status_counts,
                "gold_errors_sample": gold_errors_sample,
                "note": (
                    "All 500 examples are still present in bird_eval.jsonl "
                    "(nothing excluded). Each row also carries "
                    "metadata.gold_execution_status_at_prep so you can "
                    "filter or report on this at analysis time without "
                    "having biased which examples are in the eval set."
                ),
            },
            f,
            indent=2,
        )

    print()
    print(f"Created: {output_file}")
    print("Examples: 500")
    print(f"Gold SQL validation: {gold_status_counts}")
    print(f"Validation summary written to: {validation_summary_file}")


if __name__ == "__main__":
    main()