#!/usr/bin/env python3
"""CLI entry point for the reproducible Spider data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql_data.spider_pipeline import download_spider, prepare_spider  # noqa: E402


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download labeled Spider annotations, introspect the local SQLite "
            "databases, validate gold SQL, and emit chat-format fine-tuning JSONL."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("download", "prepare", "all"),
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/spider",
        help="Directory for checksummed source annotations.",
    )
    parser.add_argument(
        "--database-dir",
        default="milestone3/database",
        help="Spider SQLite database directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/spider",
        help="Directory for JSONL, schemas, manifests, and EDA.",
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=2.0,
        help="Per-query execution-validation timeout (default: 2.0).",
    )
    parser.add_argument(
        "--skip-query-execution",
        action="store_true",
        help="Build data without executing gold queries; structural checks still run.",
    )
    parser.add_argument(
        "--fail-on-execution-errors",
        action="store_true",
        help="Exit nonzero after writing reports if any gold query fails execution.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.query_timeout_seconds <= 0:
        raise SystemExit("--query-timeout-seconds must be positive")

    raw_dir = project_path(args.raw_dir)
    database_dir = project_path(args.database_dir)
    output_dir = project_path(args.output_dir)

    if args.command in {"download", "all"}:
        download_spider(raw_dir)

    if args.command in {"prepare", "all"}:
        report = prepare_spider(
            project_root=PROJECT_ROOT,
            raw_dir=raw_dir,
            database_dir=database_dir,
            output_dir=output_dir,
            timeout_seconds=args.query_timeout_seconds,
            execute_queries=not args.skip_query_execution,
            fail_on_execution_errors=args.fail_on_execution_errors,
        )
        print("\nSpider preparation completed.")
        print(
            "Train examples:      "
            f"{report['splits']['train']['usable_examples']} usable / "
            f"{report['splits']['train']['examples']} official"
        )
        print(
            "Validation examples: "
            f"{report['splits']['validation']['usable_examples']} usable / "
            f"{report['splits']['validation']['examples']} official"
        )
        print(f"Validation failures: {report['validation_failure_count']}")
        print(f"Database overlap:    {report['leakage_checks']['database_overlap_count']}")
        print(f"Outputs:             {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
