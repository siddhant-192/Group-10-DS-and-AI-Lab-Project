"""Reproducible preparation pipeline for Spider 1.0."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence
import urllib.request

from .schema import DatabaseSchema, inspect_database
from .validation import (
    QueryValidation,
    normalize_sql,
    open_readonly_database,
    query_features,
    validate_readonly_query,
)


SYSTEM_PROMPT = (
    "You convert natural-language questions into one read-only SQLite query. "
    "Use only the supplied database schema. Return SQL only, with no Markdown "
    "fence, explanation, or alternative query."
)


@dataclass(frozen=True)
class SourceFile:
    split: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    expected_rows: int


SPIDER_SOURCES = (
    SourceFile(
        split="train",
        filename="train.parquet",
        url=(
            "https://huggingface.co/api/datasets/xlangai/spider/"
            "parquet/spider/train/0.parquet"
        ),
        sha256="cb4b681558f6f8f428e516fb94c5a1cb19c5a0a0c153c0618c8cc4a28115d4cb",
        size_bytes=831_359,
        expected_rows=7_000,
    ),
    SourceFile(
        split="validation",
        filename="validation.parquet",
        url=(
            "https://huggingface.co/api/datasets/xlangai/spider/"
            "parquet/spider/validation/0.parquet"
        ),
        sha256="c3e2a46303899a2d4afe3f6a3a62e59f8d589f241b3cbfb52356479b1f054888",
        size_bytes=125_887,
        expected_rows=1_034,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def download_source(source: SourceFile, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.filename

    if destination.exists():
        actual_hash = sha256_file(destination)
        if actual_hash != source.sha256:
            raise RuntimeError(
                f"Existing file has the wrong checksum: {destination}\n"
                f"expected={source.sha256}\nactual={actual_hash}\n"
                "Move the file aside and re-run; it will not be overwritten automatically."
            )
        print(f"[download] verified existing {destination}")
        return destination

    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "text2sql-data-pipeline/1.0"},
    )
    print(f"[download] {source.url}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    actual_size = temporary.stat().st_size
    actual_hash = sha256_file(temporary)
    if actual_size != source.size_bytes or actual_hash != source.sha256:
        temporary.unlink()
        raise RuntimeError(
            f"Downloaded file failed integrity validation for {source.split}: "
            f"size={actual_size}, sha256={actual_hash}"
        )

    os.replace(temporary, destination)
    print(f"[download] saved and verified {destination}")
    return destination


def download_spider(raw_dir: Path) -> dict[str, Path]:
    paths = {source.split: download_source(source, raw_dir) for source in SPIDER_SOURCES}
    write_json(
        raw_dir / "source_manifest.json",
        {
            "dataset": "xlangai/spider",
            "license": "CC-BY-SA-4.0",
            "dataset_url": "https://huggingface.co/datasets/xlangai/spider",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": [asdict(source) for source in SPIDER_SOURCES],
        },
    )
    return paths


def load_annotations(path: Path, source: SourceFile) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required. Run scripts/setup_spider_data.sh first."
        ) from exc

    table = parquet.read_table(path, columns=["db_id", "question", "query"])
    raw_rows = table.to_pylist()
    if len(raw_rows) != source.expected_rows:
        raise RuntimeError(
            f"{source.split} row count changed: expected {source.expected_rows}, "
            f"found {len(raw_rows)}"
        )

    annotations: list[dict[str, str]] = []
    for index, row in enumerate(raw_rows):
        cleaned = {
            "db_id": str(row["db_id"]).strip(),
            "question": str(row["question"]).strip(),
            "query": str(row["query"]).strip(),
        }
        missing = [key for key, value in cleaned.items() if not value]
        if missing:
            raise RuntimeError(
                f"Empty required fields in {source.split} row {index}: {missing}"
            )
        annotations.append(cleaned)
    return annotations


def discover_databases(database_dir: Path) -> dict[str, Path]:
    paths = sorted(database_dir.glob("*/*.sqlite"))
    databases: dict[str, Path] = {}
    for path in paths:
        db_id = path.parent.name
        if path.stem != db_id:
            raise RuntimeError(
                f"Database directory/file mismatch: directory={db_id}, file={path.name}"
            )
        if db_id in databases:
            raise RuntimeError(f"Duplicate database ID: {db_id}")
        databases[db_id] = path
    if not databases:
        raise RuntimeError(f"No SQLite databases found under {database_dir}")
    return databases


def inspect_databases(
    databases: dict[str, Path], project_root: Path
) -> dict[str, DatabaseSchema]:
    schemas: dict[str, DatabaseSchema] = {}
    for index, (db_id, path) in enumerate(sorted(databases.items()), start=1):
        schema = inspect_database(path, relative_to=project_root)
        if schema.db_id != db_id:
            raise RuntimeError(f"Schema ID mismatch for {path}")
        schemas[db_id] = schema
        if index % 25 == 0 or index == len(databases):
            print(f"[schema] inspected {index}/{len(databases)} databases")
    return schemas


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": round(float(statistics.median(values)), 3),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def validate_queries(
    split_rows: dict[str, list[dict[str, str]]],
    databases: dict[str, Path],
    timeout_seconds: float,
    execute: bool,
) -> dict[str, list[QueryValidation]]:
    results: dict[str, list[QueryValidation]] = {}
    for split, rows in split_rows.items():
        split_results: list[QueryValidation | None] = [None] * len(rows)
        grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[row["db_id"]].append((index, row))

        completed = 0
        for db_id, group in sorted(grouped.items()):
            connection = open_readonly_database(databases[db_id]) if execute else None
            try:
                for index, row in group:
                    if execute:
                        validation = validate_readonly_query(
                            databases[db_id],
                            row["query"],
                            timeout_seconds=timeout_seconds,
                            connection=connection,
                        )
                    else:
                        validation = QueryValidation(
                            status="not_run", elapsed_ms=0.0, error=None
                        )
                    split_results[index] = validation
                    completed += 1
                    if completed % 500 == 0 or completed == len(rows):
                        print(f"[execute] {split}: {completed}/{len(rows)}")
            finally:
                if connection is not None:
                    connection.close()

        if any(result is None for result in split_results):
            raise AssertionError(f"Internal validation result gap in {split}")
        results[split] = [result for result in split_results if result is not None]
    return results


def build_example(
    split: str,
    index: int,
    annotation: dict[str, str],
    schema: DatabaseSchema,
    validation: QueryValidation,
) -> dict[str, Any]:
    ddl = schema.to_ddl()
    user_prompt = (
        "Database dialect: SQLite\n\n"
        f"Database schema:\n{ddl}\n\n"
        f"Question: {annotation['question']}"
    )
    return {
        "id": f"spider-{split}-{index:05d}",
        "dataset": "spider",
        "split": split,
        "db_id": annotation["db_id"],
        "question": annotation["question"],
        "sql": annotation["query"],
        "schema": ddl,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": annotation["query"]},
        ],
        "metadata": {
            "dialect": "sqlite",
            "database_path": schema.sqlite_path,
            "schema_sha256": hashlib.sha256(ddl.encode("utf-8")).hexdigest(),
            "query_features": query_features(annotation["query"]),
            "execution_validation": validation.to_dict(),
        },
    }


def analyze(
    split_examples: dict[str, list[dict[str, Any]]],
    schemas: dict[str, DatabaseSchema],
) -> dict[str, Any]:
    split_db_ids = {
        split: {example["db_id"] for example in examples}
        for split, examples in split_examples.items()
    }
    train_db_ids = split_db_ids.get("train", set())
    validation_db_ids = split_db_ids.get("validation", set())

    normalized_questions = {
        split: {" ".join(example["question"].casefold().split()) for example in examples}
        for split, examples in split_examples.items()
    }
    normalized_pairs = {
        split: {
            (
                example["db_id"],
                " ".join(example["question"].casefold().split()),
                normalize_sql(example["sql"]),
            )
            for example in examples
        }
        for split, examples in split_examples.items()
    }

    report: dict[str, Any] = {
        "dataset": "Spider 1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_summary": {
            "database_count": len(schemas),
            "quick_check_ok": sum(schema.quick_check == "ok" for schema in schemas.values()),
            "table_count": sum(schema.table_count for schema in schemas.values()),
            "column_count": sum(schema.column_count for schema in schemas.values()),
            "foreign_key_count": sum(schema.foreign_key_count for schema in schemas.values()),
            "tables_per_database": _distribution(
                [schema.table_count for schema in schemas.values()]
            ),
            "columns_per_database": _distribution(
                [schema.column_count for schema in schemas.values()]
            ),
            "largest_schemas_by_columns": [
                {
                    "db_id": schema.db_id,
                    "tables": schema.table_count,
                    "columns": schema.column_count,
                    "foreign_keys": schema.foreign_key_count,
                }
                for schema in sorted(
                    schemas.values(), key=lambda item: item.column_count, reverse=True
                )[:10]
            ],
        },
        "leakage_checks": {
            "database_overlap_count": len(train_db_ids & validation_db_ids),
            "database_overlap": sorted(train_db_ids & validation_db_ids),
            "normalized_question_overlap_count": len(
                normalized_questions.get("train", set())
                & normalized_questions.get("validation", set())
            ),
            "exact_pair_overlap_count": len(
                normalized_pairs.get("train", set())
                & normalized_pairs.get("validation", set())
            ),
            "unused_local_databases": sorted(
                set(schemas) - set().union(*split_db_ids.values())
            ),
        },
        "splits": {},
    }

    for split, examples in split_examples.items():
        features = [example["metadata"]["query_features"] for example in examples]
        statuses = Counter(
            example["metadata"]["execution_validation"]["status"]
            for example in examples
        )
        normalized_queries = [normalize_sql(example["sql"]) for example in examples]
        report["splits"][split] = {
            "examples": len(examples),
            "usable_examples": sum(
                example["metadata"]["execution_validation"]["status"]
                in {"ok", "not_run"}
                for example in examples
            ),
            "databases": len(split_db_ids[split]),
            "unique_questions": len(
                {" ".join(example["question"].casefold().split()) for example in examples}
            ),
            "unique_normalized_sql": len(set(normalized_queries)),
            "duplicate_sql_rows": len(normalized_queries) - len(set(normalized_queries)),
            "question_words": _distribution(
                [len(example["question"].split()) for example in examples]
            ),
            "sql_tokens": _distribution([feature["token_count"] for feature in features]),
            "schema_characters": _distribution(
                [len(example["schema"]) for example in examples]
            ),
            "prompt_estimated_tokens": _distribution(
                [
                    round(
                        (
                            len(example["messages"][0]["content"])
                            + len(example["messages"][1]["content"])
                        )
                        / 4
                    )
                    for example in examples
                ]
            ),
            "execution_status": dict(sorted(statuses.items())),
            "execution_latency_ms": _distribution(
                [
                    example["metadata"]["execution_validation"]["elapsed_ms"]
                    for example in examples
                ]
            ),
            "complexity_proxy": dict(
                sorted(Counter(feature["complexity_proxy"] for feature in features).items())
            ),
            "clauses": {
                "with_join": sum(feature["join_count"] > 0 for feature in features),
                "with_subquery": sum(feature["has_subquery"] for feature in features),
                "with_set_operation": sum(
                    feature["set_operation_count"] > 0 for feature in features
                ),
                "with_group_by": sum(feature["has_group_by"] for feature in features),
                "with_order_by": sum(feature["has_order_by"] for feature in features),
                "with_having": sum(feature["has_having"] for feature in features),
                "with_aggregate": sum(feature["aggregate_count"] > 0 for feature in features),
            },
        }

    return report


def render_markdown(report: dict[str, Any]) -> str:
    database = report["database_summary"]
    leakage = report["leakage_checks"]
    lines = [
        "# Spider Data Preparation Report",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Readiness verdict",
        "",
    ]

    validation_execution_ok = set(
        report["splits"].get("validation", {}).get("execution_status", {})
    ) <= {"ok", "not_run"}
    no_db_leakage = leakage["database_overlap_count"] == 0
    if validation_execution_ok and no_db_leakage:
        lines.append(
            "The processed data is structurally ready for a first supervised "
            "fine-tuning experiment. Non-executable training annotations were "
            "excluded from `train.jsonl`; preserve the official validation split."
        )
    else:
        lines.append(
            "The pipeline completed, but the validation findings below must be "
            "reviewed before training."
        )

    lines.extend(
        [
            "",
            "## Database inventory",
            "",
            f"- Databases: **{database['database_count']}**",
            f"- SQLite quick checks passing: **{database['quick_check_ok']}**",
            f"- Tables: **{database['table_count']}**",
            f"- Columns: **{database['column_count']}**",
            f"- Foreign keys: **{database['foreign_key_count']}**",
            "",
            "## Official splits",
            "",
            "| Split | Official examples | Usable JSONL | Databases | Unique SQL | Execution status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for split, values in report["splits"].items():
        status = ", ".join(
            f"{key}={value}" for key, value in values["execution_status"].items()
        )
        lines.append(
            f"| {split} | {values['examples']} | {values['usable_examples']} | "
            f"{values['databases']} | "
            f"{values['unique_normalized_sql']} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Rejected source annotations",
            "",
            f"- Training: **{report.get('rejected_examples', {}).get('train', 0)}**",
            f"- Validation: **{report.get('rejected_examples', {}).get('validation', 0)}**",
            "- Full SQL, questions, database IDs, and SQLite errors are preserved "
            "in `validation_failures.jsonl`.",
            "",
            "## Leakage checks",
            "",
            f"- Train/validation database overlap: **{leakage['database_overlap_count']}**",
            f"- Normalized question overlap: **{leakage['normalized_question_overlap_count']}**",
            f"- Exact `(db_id, question, SQL)` overlap: **{leakage['exact_pair_overlap_count']}**",
            f"- Local databases unused by labeled splits: "
            f"`{', '.join(leakage['unused_local_databases']) or 'none'}`",
            "",
            "## Query structure",
            "",
            "The following labels are transparent structural proxies, not Spider's "
            "official hardness labels.",
            "",
            "| Split | Simple | Moderate | Complex | Join | Subquery | Set operation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, values in report["splits"].items():
        complexity = values["complexity_proxy"]
        clauses = values["clauses"]
        lines.append(
            f"| {split} | {complexity.get('simple', 0)} | "
            f"{complexity.get('moderate', 0)} | {complexity.get('complex', 0)} | "
            f"{clauses['with_join']} | {clauses['with_subquery']} | "
            f"{clauses['with_set_operation']} |"
        )

    lines.extend(
        [
            "",
            "## Training contract",
            "",
            "Each JSONL row contains `messages` in chat fine-tuning format plus "
            "the raw `question`, `sql`, serialized `schema`, database ID, structural "
            "features, and execution-validation result. The assistant target is SQL only.",
            "",
            "Do not merge or randomly reshuffle the official splits: validation "
            "databases are deliberately unseen during training.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_spider(
    project_root: Path,
    raw_dir: Path,
    database_dir: Path,
    output_dir: Path,
    timeout_seconds: float,
    execute_queries: bool,
    fail_on_execution_errors: bool,
) -> dict[str, Any]:
    source_by_split = {source.split: source for source in SPIDER_SOURCES}
    split_rows = {
        split: load_annotations(raw_dir / source.filename, source)
        for split, source in source_by_split.items()
    }

    databases = discover_databases(database_dir)
    schemas = inspect_databases(databases, project_root)

    used_db_ids = {
        row["db_id"] for rows in split_rows.values() for row in rows
    }
    missing_databases = sorted(used_db_ids - set(databases))
    if missing_databases:
        raise RuntimeError(
            "Annotations reference missing databases: " + ", ".join(missing_databases)
        )

    train_db_ids = {row["db_id"] for row in split_rows["train"]}
    validation_db_ids = {row["db_id"] for row in split_rows["validation"]}
    database_overlap = sorted(train_db_ids & validation_db_ids)
    if database_overlap:
        raise RuntimeError(
            "Official split database leakage detected: " + ", ".join(database_overlap)
        )

    query_validations = validate_queries(
        split_rows,
        databases,
        timeout_seconds=timeout_seconds,
        execute=execute_queries,
    )

    split_examples: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for split, rows in split_rows.items():
        examples = [
            build_example(
                split,
                index,
                annotation,
                schemas[annotation["db_id"]],
                query_validations[split][index],
            )
            for index, annotation in enumerate(rows)
        ]
        split_examples[split] = examples
        for example in examples:
            validation = example["metadata"]["execution_validation"]
            if validation["status"] not in {"ok", "not_run"}:
                failures.append(
                    {
                        "id": example["id"],
                        "split": split,
                        "db_id": example["db_id"],
                        "question": example["question"],
                        "sql": example["sql"],
                        "validation": validation,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    schema_payload = {
        "schema_version": 1,
        "source": "SQLite PRAGMA introspection",
        "database_count": len(schemas),
        "databases": {
            db_id: schema.to_dict() for db_id, schema in sorted(schemas.items())
        },
    }
    write_json(output_dir / "schemas.json", schema_payload)
    usable_examples: dict[str, list[dict[str, Any]]] = {
        split: [
            example
            for example in examples
            if example["metadata"]["execution_validation"]["status"]
            in {"ok", "not_run"}
        ]
        for split, examples in split_examples.items()
    }
    for split, examples in usable_examples.items():
        written = write_jsonl(output_dir / f"{split}.jsonl", examples)
        if written != len(examples):
            raise AssertionError(f"Short write for {split}")
    write_jsonl(output_dir / "validation_failures.jsonl", failures)

    report = analyze(split_examples, schemas)
    report["validation_failure_count"] = len(failures)
    report["rejected_examples"] = {
        split: len(split_examples[split]) - len(usable_examples[split])
        for split in split_examples
    }
    report["configuration"] = {
        "query_execution_enabled": execute_queries,
        "query_timeout_seconds": timeout_seconds,
        "schema_serializer": "SQLite DDL from live PRAGMA metadata",
        "system_prompt": SYSTEM_PROMPT,
    }
    write_json(output_dir / "eda_report.json", report)
    write_text(output_dir / "EDA.md", render_markdown(report))

    output_files = [
        "schemas.json",
        "train.jsonl",
        "validation.jsonl",
        "validation_failures.jsonl",
        "eda_report.json",
        "EDA.md",
    ]
    manifest = {
        "dataset": "Spider 1.0",
        "license": "CC-BY-SA-4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "source_files": {
            source.split: {
                "path": str((raw_dir / source.filename).relative_to(project_root)),
                "sha256": sha256_file(raw_dir / source.filename),
                "rows": len(split_rows[source.split]),
                "url": source.url,
            }
            for source in SPIDER_SOURCES
        },
        "database_dir": str(database_dir.relative_to(project_root)),
        "outputs": {
            filename: {
                "path": str((output_dir / filename).relative_to(project_root)),
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": sha256_file(output_dir / filename),
            }
            for filename in output_files
        },
        "summary": {
            "source_train_examples": len(split_examples["train"]),
            "source_validation_examples": len(split_examples["validation"]),
            "usable_train_examples": len(usable_examples["train"]),
            "usable_validation_examples": len(usable_examples["validation"]),
            "validation_failures": len(failures),
            "database_overlap": len(database_overlap),
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    if fail_on_execution_errors and failures:
        raise RuntimeError(
            f"Found {len(failures)} execution-validation failures. "
            f"Review {output_dir / 'validation_failures.jsonl'}"
        )

    return report
