#!/usr/bin/env python3
"""Re-select stored SQL candidates locally without regenerating model output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_text2sql_models import (  # noqa: E402
    ORDER_BY,
    execute_query,
    macsql_postprocess,
    resolve_database,
    select_execution_consensus,
    select_value_aware_voting,
    strip_distinct_sql,
)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--selection",
        choices=("execution-consensus", "value-aware-voting"),
        required=True,
    )
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = {str(row["id"]): row for row in jsonl(args.validation)}
    output: list[dict] = []
    for prediction in jsonl(args.predictions):
        row = validation[str(prediction["id"])]
        candidates = prediction.get("candidates") or []
        if not candidates:
            raise ValueError(f"prediction {prediction['id']} has no stored candidates")
        database = resolve_database(args.project_root.resolve(), row)
        order_sensitive = bool(ORDER_BY.search(str(row["sql"])))
        results = [
            execute_query(
                database,
                strip_distinct_sql(macsql_postprocess(str(candidate["predicted_sql"]))),
                args.query_timeout,
                order_sensitive,
            )
            for candidate in candidates
        ]
        if args.selection == "value-aware-voting":
            selected, votes = select_value_aware_voting(results)
        else:
            selected, votes = select_execution_consensus(results)
        chosen = candidates[selected]
        updated = dict(prediction)
        updated.update(
            {
                "raw_prediction": chosen["raw_prediction"],
                "predicted_sql": chosen["predicted_sql"],
                "selected_candidate_index": selected,
                "execution_consensus_votes": votes,
                "candidate_selection": args.selection,
            }
        )
        output.append(updated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"predictions": len(output), "selection": args.selection}))


if __name__ == "__main__":
    main()
