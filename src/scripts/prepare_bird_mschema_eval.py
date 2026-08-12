#!/usr/bin/env python3
"""
Prepare BIRD Mini-Dev evaluation data using the SAME M-Schema prompt format
that checkpoint 375 was actually trained on (per Milestone 4: "System +
M-Schema + question -> Gold SQL"), instead of plain DDL.

This matters because the original prepare_bird_eval.py built plain
"CREATE TABLE (...)" DDL strings, which checkpoint 375 never saw during
training -- it was trained exclusively on M-Schema-formatted prompts
(build_mschema_sft_package.py -> render_mschema()). Evaluating with a
prompt format the model was never trained on confounds "cross-benchmark
generalization" with "unfamiliar prompt format," making the BIRD numbers
harder to interpret cleanly.

This script:
  - Reuses render_mschema() directly from build_xiyan_mschema_eval_data.py
    (the actual function used to build checkpoint 375's training data) so
    there is zero risk of the M-Schema format drifting from what was
    actually trained on.
  - Replicates the exact same DDL-fallback-for-oversized-schema behavior
    used during training (default 10,000 char cap), so large BIRD
    databases (e.g. card_games, codebase_community) are handled exactly
    the way training handled oversized Spider databases.
  - Reuses the same prompt() template used in build_mschema_sft_package.py,
    verbatim, including leaving 【Evidence】 blank (matching how Spider
    training examples were built, since Spider has no evidence field --
    this isolates the M-Schema format fix as the only change, without
    also introducing BIRD's evidence hints as a second simultaneous
    variable).
  - Keeps the validity-driven gold SQL resolution from the earlier BIRD
    fix (tries the raw gold line first, only strips a trailing db_id
    suffix if that's what makes it valid) -- unrelated to this fix, but
    still necessary and unaffected by it.

Usage:
    python prepare_bird_mschema_eval.py \
        --bird-root /content/MINIDEV \
        --output-dir evidence/milestone5/bird \
        --render-script-dir src/scripts
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare BIRD Mini-Dev data using checkpoint 375's actual M-Schema training format."
    )

    parser.add_argument("--bird-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/milestone5/bird"))
    parser.add_argument(
        "--render-script-dir",
        type=Path,
        required=True,
        help="Directory containing build_xiyan_mschema_eval_data.py (e.g. src/scripts)",
    )
    parser.add_argument("--gold-timeout", type=float, default=15.0)
    parser.add_argument(
        "--max-mschema-chars",
        type=int,
        default=10_000,
        help="Same cap used in build_mschema_sft_package.py -- falls back to DDL above this size.",
    )
    parser.add_argument("--mschema-examples", type=int, default=3,
                         help="Number of example values per column, matching training default.")
    parser.add_argument(
        "--include-evidence",
        action="store_true",
        default=False,
        help=(
            "If set, insert BIRD's natural-language evidence hint into the "
            "prompt's 【Evidence】 section (BIRD-intended usage). Default "
            "(off) leaves it blank, matching how checkpoint 375 was trained "
            "on Spider (which has no evidence field) -- this is the setting "
            "used to produce the report's primary 25.8%% result. Enabling "
            "this changes the output filename and does not overwrite the "
            "existing bird_eval_mschema.jsonl, preserving reproducibility "
            "of the original result."
        ),
    )

    return parser.parse_args()


def find_database(database_dir, db_id):
    db_dir = database_dir / db_id
    candidates = [db_dir / f"{db_id}.sqlite", db_dir / f"{db_id}.db"]
    for path in candidates:
        if path.exists():
            return path
    for pattern in ("*.sqlite", "*.db"):
        matches = sorted(db_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No SQLite database found for db_id={db_id} in {db_dir}")


def get_ddl_schema(db_path):
    """Plain DDL fallback, same shape as the original prepare_bird_eval.py --
    used only when the rendered M-Schema exceeds --max-mschema-chars, exactly
    mirroring training-time behavior for oversized databases."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        schema_parts = []
        for (table_name,) in tables:
            columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            column_names = [row[1] for row in columns]
            schema_parts.append(f"CREATE TABLE {table_name} ({', '.join(column_names)});")
        return "\n".join(schema_parts)
    finally:
        connection.close()


def validate_gold_sql(db_path, sql, timeout):
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
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "elapsed": time.monotonic() - t0}


def resolve_gold_sql(raw_line, db_id, db_path, timeout):
    """Validity-driven gold SQL resolution (unchanged from the earlier BIRD fix):
    tries the line as-written first, only strips a trailing db_id suffix if
    that's what makes it valid -- avoids whitespace-count guessing, which
    can't reliably distinguish real corruption from queries that legitimately
    end in a table name matching the db_id."""
    full_candidate = raw_line.strip()
    full_check = validate_gold_sql(db_path, full_candidate, timeout)
    if full_check["status"] == "ok":
        return full_candidate, full_check

    stripped_candidate = re.sub(rf"\s+{re.escape(db_id)}\s*$", "", raw_line).strip()
    if stripped_candidate != full_candidate:
        stripped_check = validate_gold_sql(db_path, stripped_candidate, timeout)
        if stripped_check["status"] == "ok":
            return stripped_candidate, stripped_check
        return stripped_candidate, stripped_check

    return full_candidate, full_check


def build_prompt(schema, question, evidence=None):
    """Verbatim copy of prompt() from build_mschema_sft_package.py -- the
    EXACT template checkpoint 375 was trained on. 【Evidence】 is left blank
    by default, matching how Spider training examples were built (Spider
    has no evidence field). When evidence is provided (--include-evidence),
    it is inserted into that section -- BIRD's intended usage pattern, but
    a deliberate departure from the training-matched default.

    IMPORTANT: when evidence is None/empty, this must produce a BYTE-
    IDENTICAL string to the original template (no extra newlines), so the
    default (--include-evidence unset) reproduces the reported 25.8%
    result exactly.
    """
    evidence_text = evidence.strip() if evidence else ""
    evidence_block = f"{evidence_text}\n\n" if evidence_text else "\n"
    return (
        "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
        f"【Schema】\n{schema}\n\n"
        f"【Question】\n{question}\n\n"
        f"【Evidence】\n{evidence_block}"
        "Please read and understand the database schema carefully, and generate an executable SQL based "
        "on the user's question and evidence. The generated SQL is protected by ```sql and ```."
    )


def main():
    args = parse_args()

    # Reuse render_mschema() DIRECTLY from the repo's own training-data
    # rendering script -- guarantees zero drift from what was actually
    # used to build checkpoint 375's training examples.
    sys.path.insert(0, str(args.render_script_dir.resolve()))
    from build_xiyan_mschema_eval_data import render_mschema  # noqa: E402

    bird_root = args.bird_root.resolve()
    output_dir = args.output_dir.resolve()

    questions_file = bird_root / "mini_dev_sqlite.json"
    gold_file = bird_root / "mini_dev_sqlite_gold.sql"
    database_dir = bird_root / "dev_databases"

    for path in (questions_file, gold_file, database_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required BIRD file/directory not found: {path}")

    print(f"BIRD root: {bird_root}")
    print(f"Output directory: {output_dir}")

    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)

    with open(gold_file, encoding="utf-8") as f:
        gold_lines = [line.rstrip("\n") for line in f]

    if len(questions) != 500:
        raise RuntimeError(f"Expected 500 BIRD questions, got {len(questions)}")
    if len(gold_lines) != 500:
        raise RuntimeError(f"Expected 500 gold SQL lines, got {len(gold_lines)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Separate output filename when evidence is included -- NEVER overwrites
    # bird_eval_mschema.jsonl, so the original reported 25.8% result stays
    # exactly reproducible regardless of whether this flag is used later.
    output_filename = (
        "bird_eval_mschema_with_evidence.jsonl" if args.include_evidence
        else "bird_eval_mschema.jsonl"
    )
    validation_filename = (
        "gold_validation_mschema_with_evidence.json" if args.include_evidence
        else "gold_validation_mschema.json"
    )
    output_file = output_dir / output_filename
    validation_summary_file = output_dir / validation_filename

    gold_status_counts = {"ok": 0, "error": 0}
    gold_errors_sample = []
    schema_cache = {}
    ddl_fallback_databases = set()
    ddl_fallback_rows = 0

    with open(output_file, "w", encoding="utf-8") as out:

        for index, (question, gold_line) in enumerate(zip(questions, gold_lines)):
            db_id = question["db_id"]
            db_path = find_database(database_dir, db_id)

            # Render M-Schema once per database (cached), with the SAME
            # oversized-schema DDL fallback used during training.
            if db_id not in schema_cache:
                mschema = render_mschema(db_path, db_id, args.mschema_examples)
                if len(mschema) > args.max_mschema_chars:
                    ddl_fallback_databases.add(db_id)
                    schema_cache[db_id] = {
                        "schema": get_ddl_schema(db_path),
                        "prompt_variant": "project_ddl_fallback_large_mschema",
                    }
                else:
                    schema_cache[db_id] = {
                        "schema": mschema,
                        "prompt_variant": "xiyan_official_mschema_english",
                    }

            schema_info = schema_cache[db_id]
            schema = schema_info["schema"]
            if schema_info["prompt_variant"] == "project_ddl_fallback_large_mschema":
                ddl_fallback_rows += 1

            gold_sql, gold_check = resolve_gold_sql(gold_line, db_id, db_path, args.gold_timeout)

            gold_status_counts[gold_check["status"]] = gold_status_counts.get(gold_check["status"], 0) + 1
            if gold_check["status"] != "ok" and len(gold_errors_sample) < 20:
                gold_errors_sample.append({"id": index, "db_id": db_id, "error": gold_check["error"]})

            question_text = question["question"]
            evidence_text = question.get("evidence", "") if args.include_evidence else None
            prompt_text = build_prompt(schema, question_text, evidence=evidence_text)

            row = {
                "id": index,
                "db_id": db_id,
                "question": question_text,
                "sql": gold_sql,
                "schema": schema,
                "metadata": {
                    "database_path": str(db_path),
                    "prompt_variant": schema_info["prompt_variant"],
                    "gold_execution_status_at_prep": gold_check["status"],
                    "gold_execution_error_at_prep": gold_check["error"],
                },
                "messages": [
                    {"role": "user", "content": prompt_text},
                ],
            }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (index + 1) % 50 == 0:
                print(
                    f"Prepared {index + 1}/500  "
                    f"(gold ok={gold_status_counts.get('ok', 0)}, "
                    f"error={gold_status_counts.get('error', 0)}, "
                    f"ddl_fallback_rows={ddl_fallback_rows})"
                )

    with open(validation_summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_examples": 500,
                "gold_status_counts": gold_status_counts,
                "gold_errors_sample": gold_errors_sample,
                "databases_using_mschema": len(schema_cache) - len(ddl_fallback_databases),
                "databases_using_ddl_fallback": sorted(ddl_fallback_databases),
                "rows_using_ddl_fallback": ddl_fallback_rows,
                "note": (
                    "Schema format now matches checkpoint 375's actual training "
                    "format (M-Schema, with the same DDL fallback for oversized "
                    "schemas used during training). All 500 examples retained; "
                    "nothing excluded."
                ),
            },
            f,
            indent=2,
        )

    print()
    print(f"Created: {output_file}")
    print(f"Gold SQL validation: {gold_status_counts}")
    print(f"Databases using M-Schema: {len(schema_cache) - len(ddl_fallback_databases)}")
    print(f"Databases using DDL fallback (oversized): {sorted(ddl_fallback_databases)}")
    print(f"Validation summary written to: {validation_summary_file}")


if __name__ == "__main__":
    main()