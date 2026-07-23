#!/usr/bin/env python3
"""Sequential, resumable evaluation for configured text-to-SQL models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys
import tarfile
import time
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import quote

import sqlglot
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "text2sql_eval_models.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "text2sql-eval" / "download_manifest.json"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "zero-shot-eval" / "results"
SYSTEM_FALLBACK = (
    "You convert natural-language questions into one read-only SQLite query. "
    "Use only the supplied database schema. Return SQL only, with no Markdown "
    "fence, explanation, or alternative query."
)
READ_ONLY_PREFIX = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
SQL_START = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
FENCED_SQL = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
MAX_RESULT_ROWS = 100_000


def sqlite_actions(*names: str) -> frozenset[int]:
    return frozenset(
        int(value)
        for name in names
        if (value := getattr(sqlite3, name, None)) is not None
    )


DENIED_ACTIONS = sqlite_actions(
    "SQLITE_ALTER_TABLE",
    "SQLITE_ANALYZE",
    "SQLITE_ATTACH",
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_VTABLE",
    "SQLITE_DELETE",
    "SQLITE_DETACH",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_DROP_VTABLE",
    "SQLITE_INSERT",
    "SQLITE_PRAGMA",
    "SQLITE_REINDEX",
    "SQLITE_TRANSACTION",
    "SQLITE_UPDATE",
)


@dataclass(frozen=True)
class QueryResult:
    status: str
    rows: tuple[str, ...] | None
    column_count: int | None
    elapsed_ms: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models" / "text2sql-eval")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-source",
        choices=("auto", "local", "huggingface"),
        default="auto",
        help="Use uploaded directories, pinned Hugging Face revisions, or prefer local when present.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/huggingface-cache"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Generate this many SQL candidates per question; values above one use sampling.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--candidate-selection",
        choices=("execution-consensus", "value-aware-voting"),
        default="execution-consensus",
        help="How to choose among sampled executable candidates.",
    )
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="Load one local PEFT adapter on the selected pinned base model.",
    )
    parser.add_argument(
        "--adapter-label",
        help="Output slug used for an adapter evaluation (required with --adapter-dir).",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("text2sql-eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_status(path: Path, **values: Any) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    atomic_json(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def load_specs(config_path: Path, manifest_path: Path, selected: set[str] | None) -> list[dict[str, Any]]:
    specs = json.loads(config_path.read_text(encoding="utf-8"))["models"]
    manifest_by_slug: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest_by_slug = {
            str(item["slug"]): item
            for item in json.loads(manifest_path.read_text(encoding="utf-8"))["models"]
        }
    merged = []
    for spec in specs:
        item = {**spec, **manifest_by_slug.get(str(spec["slug"]), {})}
        if not selected or str(item["slug"]) in selected:
            merged.append(item)
    if selected:
        missing = selected - {str(item["slug"]) for item in merged}
        if missing:
            raise ValueError(f"Unknown model slug(s): {', '.join(sorted(missing))}")
    if not merged:
        raise ValueError("No models selected")
    return merged


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages") or []
    inputs = [message for message in messages if message.get("role") != "assistant"]
    if inputs:
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in inputs]
    user = f"Database dialect: SQLite\n\nDatabase schema:\n{row['schema']}\n\nQuestion: {row['question']}"
    return [{"role": "system", "content": SYSTEM_FALLBACK}, {"role": "user", "content": user}]


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = prompt_messages(row)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        # Some older code-model templates accept user/assistant roles only. Keep
        # the exact same instruction and question, but fold them into one user turn.
        if not any(term in str(exc).lower() for term in ("role", "system", "alternat")):
            raise
        combined = "\n\n".join(str(message["content"]) for message in messages)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": combined}], tokenize=False, add_generation_prompt=True
        )


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split()).lower()


def extract_sql(raw: str) -> str:
    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    fenced = FENCED_SQL.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = SQL_START.search(text)
    if start:
        text = text[start.start() :]
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    return text.strip()


def canonical_sql(sql: str) -> tuple[str | None, str | None]:
    try:
        expressions = sqlglot.parse(sql, read="sqlite")
        if len(expressions) != 1:
            return None, f"expected one SQL statement, parsed {len(expressions)}"
        return expressions[0].sql(dialect="sqlite", normalize=True, pretty=False), None
    except Exception as exc:  # sqlglot uses several ParseError subclasses
        return None, str(exc)


def readonly_uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()), safe='/')}?mode=ro&immutable=1"


@contextmanager
def readonly_connection(path: Path, timeout_seconds: float) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(readonly_uri(path), uri=True)
    deadline = time.monotonic() + timeout_seconds

    def authorizer(
        action: int,
        _arg1: str | None,
        _arg2: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        return sqlite3.SQLITE_DENY if action in DENIED_ACTIONS else sqlite3.SQLITE_OK

    def progress() -> int:
        return 1 if time.monotonic() > deadline else 0

    connection.set_authorizer(authorizer)
    connection.set_progress_handler(progress, 10_000)
    try:
        yield connection
    finally:
        connection.set_progress_handler(None, 0)
        connection.set_authorizer(None)
        connection.close()


def json_cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def execute_query(path: Path, sql: str, timeout_seconds: float, order_sensitive: bool) -> QueryResult:
    started = time.monotonic()
    if not READ_ONLY_PREFIX.match(sql):
        return QueryResult("unsafe", None, None, 0.0, "query does not begin with SELECT or WITH")
    try:
        with readonly_connection(path, timeout_seconds) as connection:
            cursor = connection.execute(sql)
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
            if len(rows) > MAX_RESULT_ROWS:
                return QueryResult(
                    "too_many_rows", None, len(cursor.description or ()),
                    round((time.monotonic() - started) * 1000, 3),
                    f"result exceeded {MAX_RESULT_ROWS} rows",
                )
            serialized = tuple(
                json.dumps([json_cell(cell) for cell in row], sort_keys=True, default=str)
                for row in rows
            )
            if not order_sensitive:
                serialized = tuple(sorted(serialized))
            return QueryResult(
                "ok", serialized, len(cursor.description or ()),
                round((time.monotonic() - started) * 1000, 3), None,
            )
    except sqlite3.DatabaseError as exc:
        message = str(exc)
        status = "timeout" if "interrupted" in message.lower() else "error"
        return QueryResult(status, None, None, round((time.monotonic() - started) * 1000, 3), message)


def resolve_database(project_root: Path, row: dict[str, Any]) -> Path:
    stored = Path(str(row.get("metadata", {}).get("database_path", "")))
    candidate = stored if stored.is_absolute() else project_root / stored
    if candidate.exists():
        return candidate
    fallback = project_root / "milestone3" / "database" / str(row["db_id"]) / f"{row['db_id']}.sqlite"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Database not found for {row['id']}: tried {candidate} and {fallback}")


def execution_match(
    project_root: Path,
    row: dict[str, Any],
    predicted_sql: str,
    timeout_seconds: float,
    gold_cache: dict[tuple[str, str, bool], QueryResult],
) -> tuple[bool, QueryResult, QueryResult]:
    database = resolve_database(project_root, row)
    gold_sql = str(row["sql"])
    order_sensitive = bool(ORDER_BY.search(gold_sql))
    cache_key = (str(database), gold_sql, order_sensitive)
    gold = gold_cache.get(cache_key)
    if gold is None:
        gold = execute_query(database, gold_sql, timeout_seconds, order_sensitive)
        gold_cache[cache_key] = gold
    predicted = execute_query(database, predicted_sql, timeout_seconds, order_sensitive)
    matched = (
        gold.status == "ok"
        and predicted.status == "ok"
        and gold.column_count == predicted.column_count
        and gold.rows == predicted.rows
    )
    return matched, gold, predicted


def strip_distinct_sql(sql: str) -> str:
    """Remove DISTINCT tokens outside quoted strings, matching MAC-SQL's EX setup.

    The published MAC-SQL/FINER evaluator removes DISTINCT from both gold and
    predicted SQL before execution.  A small lexical scanner avoids changing a
    string literal such as ``'distinct'`` while keeping this evaluator free of
    another runtime dependency.
    """
    output: list[str] = []
    cursor = 0
    quote_character: str | None = None
    bracket_quote = False
    while cursor < len(sql):
        character = sql[cursor]
        if quote_character is not None:
            output.append(character)
            if character == quote_character:
                if cursor + 1 < len(sql) and sql[cursor + 1] == quote_character:
                    output.append(sql[cursor + 1])
                    cursor += 1
                else:
                    quote_character = None
            cursor += 1
            continue
        if bracket_quote:
            output.append(character)
            if character == "]":
                bracket_quote = False
            cursor += 1
            continue
        if character in ("'", '"', "`"):
            quote_character = character
            output.append(character)
            cursor += 1
            continue
        if character == "[":
            bracket_quote = True
            output.append(character)
            cursor += 1
            continue
        if character.isalpha() or character == "_":
            end = cursor + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            token = sql[cursor:end]
            if token.lower() != "distinct":
                output.append(token)
            cursor = end
            continue
        output.append(character)
        cursor += 1
    return "".join(output)


def macsql_postprocess(sql: str) -> str:
    """Apply the upstream evaluator's three spaced-operator repairs."""
    return sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")


def _decoded_result_rows(result: QueryResult) -> list[tuple[Any, ...]] | None:
    """Decode JSON rows while preserving SQLite/Python numeric equality."""
    if result.status != "ok" or result.rows is None:
        return None
    decoded: list[tuple[Any, ...]] = []
    for serialized_row in result.rows:
        try:
            values = json.loads(serialized_row)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(values, list):
            values = [values]
        cells: list[Any] = []
        for value in values:
            if isinstance(value, dict) and set(value) == {"bytes_hex"}:
                try:
                    cells.append(bytes.fromhex(str(value["bytes_hex"])))
                    continue
                except ValueError:
                    pass
            if isinstance(value, (dict, list)):
                value = ("json", json.dumps(value, sort_keys=True, default=str))
            cells.append(value)
        decoded.append(tuple(cells))
    return decoded


def macsql_result_equal(
    gold: QueryResult,
    predicted: QueryResult,
    order_matters: bool,
) -> bool:
    """Compare denotations using the official MAC-SQL Spider EX semantics.

    In contrast to this project's deliberately strict metric, MAC-SQL permits
    one global permutation of result columns.  Rows use bag semantics unless
    the gold query contains ORDER BY.  This implementation is deterministic;
    the upstream code samples rows only to prune the same permutation search.
    """
    gold_rows = _decoded_result_rows(gold)
    predicted_rows = _decoded_result_rows(predicted)
    if gold_rows is None or predicted_rows is None:
        return False
    if not gold_rows and not predicted_rows:
        return True
    if len(gold_rows) != len(predicted_rows) or not gold_rows or not predicted_rows:
        return False
    column_count = len(gold_rows[0])
    if len(predicted_rows[0]) != column_count:
        return False
    if any(len(row) != column_count for row in gold_rows + predicted_rows):
        return False

    # Equivalent tables must have the same bag of values within each row even
    # before the single global column permutation is determined.
    sort_key = lambda value: str(value) + str(type(value))
    unordered_gold = [tuple(sorted(row, key=sort_key)) for row in gold_rows]
    unordered_predicted = [tuple(sorted(row, key=sort_key)) for row in predicted_rows]
    if order_matters:
        if unordered_gold != unordered_predicted:
            return False
    elif Counter(unordered_gold) != Counter(unordered_predicted):
        return False

    gold_column_values = [
        frozenset(row[column] for row in gold_rows) for column in range(column_count)
    ]
    predicted_column_values = [
        frozenset(row[column] for row in predicted_rows) for column in range(column_count)
    ]
    choices = [
        [
            predicted_column
            for predicted_column, values in enumerate(predicted_column_values)
            if values == gold_values
        ]
        for gold_values in gold_column_values
    ]
    if any(not candidates for candidates in choices):
        return False

    target = gold_rows if order_matters else Counter(gold_rows)
    column_order = sorted(range(column_count), key=lambda column: len(choices[column]))
    permutation = [-1] * column_count

    def search(depth: int, used: set[int]) -> bool:
        if depth == column_count:
            permuted = [
                tuple(row[permutation[column]] for column in range(column_count))
                for row in predicted_rows
            ]
            return permuted == target if order_matters else Counter(permuted) == target
        gold_column = column_order[depth]
        for predicted_column in choices[gold_column]:
            if predicted_column in used:
                continue
            permutation[gold_column] = predicted_column
            used.add(predicted_column)
            if search(depth + 1, used):
                return True
            used.remove(predicted_column)
        permutation[gold_column] = -1
        return False

    return search(0, set())


def macsql_execution_match(
    project_root: Path,
    row: dict[str, Any],
    predicted_sql: str,
    timeout_seconds: float,
    gold_cache: dict[tuple[str, str, bool], QueryResult],
) -> tuple[bool, QueryResult, QueryResult]:
    """Execute and score one prediction like MAC-SQL/FINER (plug_value=False)."""
    database = resolve_database(project_root, row)
    gold_sql = strip_distinct_sql(macsql_postprocess(str(row["sql"])))
    prediction = strip_distinct_sql(macsql_postprocess(predicted_sql))
    order_sensitive = bool(ORDER_BY.search(gold_sql))
    cache_key = (str(database), gold_sql, order_sensitive)
    gold = gold_cache.get(cache_key)
    if gold is None:
        gold = execute_query(database, gold_sql, timeout_seconds, order_sensitive)
        gold_cache[cache_key] = gold
    predicted = execute_query(database, prediction, timeout_seconds, order_sensitive)
    return macsql_result_equal(gold, predicted, order_sensitive), gold, predicted


def execution_signature(result: QueryResult) -> tuple[int | None, tuple[str, ...]] | None:
    """Return a hashable result signature without consulting the gold query."""
    if result.status != "ok" or result.rows is None:
        return None
    return result.column_count, result.rows


def select_execution_consensus(results: Sequence[QueryResult]) -> tuple[int, int]:
    """Select the earliest candidate in the largest executable-result cluster.

    Failed candidates never outvote an executable candidate. If every candidate
    fails, return the first candidate as an explicit deterministic fallback.
    """
    signatures = [execution_signature(result) for result in results]
    counts = Counter(signature for signature in signatures if signature is not None)
    if not counts:
        return 0, 0
    winning_signature, votes = counts.most_common(1)[0]
    return signatures.index(winning_signature), votes


def value_aware_signature(result: QueryResult) -> tuple[tuple[str, ...], ...] | None:
    """Group successful results by values, ignoring headers and column order."""
    if result.status != "ok" or result.rows is None:
        return None
    rows: set[tuple[str, ...]] = set()
    for serialized_row in result.rows:
        try:
            values = json.loads(serialized_row)
        except (TypeError, json.JSONDecodeError):
            values = [serialized_row]
        if not isinstance(values, list):
            values = [values]
        rows.add(tuple(sorted(str(value) for value in values)))
    return tuple(sorted(rows))


def value_aware_result_is_all_zero(signature: tuple[tuple[str, ...], ...]) -> bool:
    values = [value for row in signature for value in row]
    if not values:
        return False
    try:
        return all(abs(float(value.rstrip("%").strip())) < 1e-12 for value in values)
    except ValueError:
        return False


def select_value_aware_voting(results: Sequence[QueryResult]) -> tuple[int, int]:
    """FINER-style voting, skipping empty and all-zero executable clusters."""
    signatures = [value_aware_signature(result) for result in results]
    executable = [signature for signature in signatures if signature is not None]
    if not executable:
        return 0, 0
    preferred = [
        signature
        for signature in executable
        if signature and not value_aware_result_is_all_zero(signature)
    ]
    eligible = preferred or executable
    counts = Counter(eligible)
    winning_signature, votes = max(
        counts.items(),
        key=lambda item: (item[1], json.dumps(item[0], sort_keys=True)),
    )
    return signatures.index(winning_signature), votes


def finer_published_signature(result: QueryResult) -> str | None:
    """Match FINER-SQL's published 200-character execution-value grouping key."""
    if result.status != "ok" or result.rows is None:
        return None
    row_strings: set[str] = set()
    for serialized_row in result.rows:
        try:
            values = json.loads(serialized_row)
        except (TypeError, json.JSONDecodeError):
            values = [serialized_row]
        if not isinstance(values, list):
            values = [values]
        row_strings.add("|".join(sorted(str(value) for value in values)))
    return ";".join(sorted(row_strings))[:200]


def finer_published_signature_is_all_zero(signature: str) -> bool:
    values = [value.strip() for value in signature.split(";") if value.strip()]
    numbers: list[float] = []
    for value in values:
        candidate = value[:-1].strip() if value.endswith("%") else value
        if not re.fullmatch(r"[\s\-+]?(?:\d+(?:\.\d+)?)", candidate):
            continue
        numbers.append(float(candidate))
    return bool(numbers) and all(abs(value) < 1e-12 for value in numbers)


def select_finer_published_vav(results: Sequence[QueryResult]) -> tuple[int, int]:
    """Reproduce FINER-SQL's published empty/zero-filtered VAV selector."""
    signatures = [finer_published_signature(result) for result in results]
    executable = [signature for signature in signatures if signature is not None]
    if not executable:
        return 0, 0
    preferred = [
        signature
        for signature in executable
        if signature and not finer_published_signature_is_all_zero(signature)
    ]
    eligible = preferred or executable
    counts = Counter(eligible)
    winning_signature, votes = max(counts.items(), key=lambda item: (item[1], item[0]))
    return signatures.index(winning_signature), votes


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def summarize(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    keys = (
        "raw_exact_match",
        "normalized_exact_match",
        "canonical_exact_match",
        "syntax_valid",
        "execution_match",
        "format_compliant",
        "input_truncated",
        "candidate_oracle_match",
    )
    counts = {key: sum(bool(row.get(key)) for row in predictions) for key in keys}
    breakdown: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[str(row.get("complexity", "unknown"))].append(row)
    for name, rows in sorted(groups.items()):
        breakdown[name] = {
            "examples": len(rows),
            "execution_accuracy_pct": percentage(sum(bool(row.get("execution_match")) for row in rows), len(rows)),
            "syntax_valid_pct": percentage(sum(bool(row.get("syntax_valid")) for row in rows), len(rows)),
            "normalized_exact_match_pct": percentage(
                sum(bool(row.get("normalized_exact_match")) for row in rows), len(rows)
            ),
        }
    latencies = [float(row["generation_ms_per_example"]) for row in predictions if row.get("generation_ms_per_example") is not None]
    return {
        "examples": total,
        **{f"{key}_count": value for key, value in counts.items()},
        **{f"{key}_pct": percentage(value, total) for key, value in counts.items()},
        "prediction_execution_status": dict(Counter(str(row.get("prediction_execution_status")) for row in predictions)),
        "mean_generation_ms_per_example": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "complexity_breakdown": breakdown,
    }


def model_location(spec: dict[str, Any], args: argparse.Namespace) -> tuple[str, bool]:
    local = args.model_root.resolve() / str(spec.get("base_slug", spec["slug"]))
    if args.model_source == "local" or (args.model_source == "auto" and local.exists()):
        if not local.exists():
            raise FileNotFoundError(f"Local model directory not found: {local}")
        return str(local), True
    return str(spec["repo_id"]), False


def load_model_and_tokenizer(spec: dict[str, Any], args: argparse.Namespace, logger: logging.Logger) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source, local_only = model_location(spec, args)
    revision = None if local_only else spec.get("revision")
    if not local_only and bool(spec.get("sequential_download", False)):
        from huggingface_hub import snapshot_download

        delays = (0, 15, 45, 90)
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                logger.warning(
                    "Retrying sequential snapshot download for %s in %d seconds (attempt %d/%d).",
                    spec["slug"],
                    delay,
                    attempt,
                    len(delays),
                )
                time.sleep(delay)
            try:
                source = snapshot_download(
                    repo_id=source,
                    revision=revision,
                    cache_dir=str(args.cache_dir),
                    max_workers=1,
                )
                local_only = True
                revision = None
                logger.info("Downloaded %s sequentially to %s", spec["slug"], source)
                break
            except Exception:
                if attempt == len(delays):
                    raise
                logger.exception(
                    "Sequential snapshot download attempt %d/%d failed for %s.",
                    attempt,
                    len(delays),
                    spec["slug"],
                )
    common = {
        "revision": revision,
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        "local_files_only": local_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {
        **common,
        "cache_dir": None if local_only else str(args.cache_dir),
        "torch_dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }
    attention = "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(source, attn_implementation="sdpa", **model_kwargs)
    except (TypeError, ValueError, ImportError) as exc:
        message = str(exc).lower()
        if not any(term in message for term in ("sdpa", "attn", "attention", "unexpected keyword")):
            raise
        logger.warning("SDPA is unavailable for %s (%s); retrying with its native attention.", spec["slug"], exc)
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        attention = "native"
    if args.adapter_dir is not None:
        from peft import PeftModel

        adapter_config_path = args.adapter_dir / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        expected_revision = str(spec.get("revision") or "")
        if str(adapter_config.get("revision") or "") != expected_revision:
            raise ValueError(
                f"Adapter revision mismatch: expected pinned base {expected_revision}, "
                f"found {adapter_config.get('revision')!r} in {adapter_config_path}"
            )
        configured_base = str(adapter_config.get("base_model_name_or_path") or "")
        if configured_base and configured_base != str(spec["repo_id"]):
            raise ValueError(
                f"Adapter base mismatch: expected {spec['repo_id']}, found {configured_base}"
            )
        model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=False)
        logger.info("Loaded PEFT adapter %s on pinned base %s", args.adapter_dir, expected_revision)
    model.eval()
    return model, tokenizer, attention


def cuda_memory() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def append_predictions(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()


def evaluate_model(
    spec: dict[str, Any],
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    logger: logging.Logger,
    status_path: Path,
) -> dict[str, Any]:
    import torch

    slug = str(spec["slug"])
    output_dir = args.output_dir / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    completed_rows = read_jsonl(prediction_path) if prediction_path.exists() else []
    completed_ids = {str(row["id"]) for row in completed_rows}
    pending = [row for row in examples if str(row["id"]) not in completed_ids]
    logger.info("%s: %d completed, %d pending", slug, len(completed_ids), len(pending))
    update_status(
        status_path,
        phase="loading_model",
        current_model=slug,
        completed_examples=len(completed_ids),
        total_examples=len(examples),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, tokenizer, attention = load_model_and_tokenizer(spec, args, logger)
    load_seconds = round(time.monotonic() - started, 3)
    logger.info("%s loaded in %.1fs with %s attention; CUDA=%s", slug, load_seconds, attention, cuda_memory())

    rendered: list[tuple[int, dict[str, Any], str]] = []
    for row in pending:
        prompt = render_prompt(tokenizer, row)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count > args.max_input_tokens:
            raise ValueError(
                f"{row['id']} requires {token_count} input tokens for {slug}, exceeding "
                f"--max-input-tokens={args.max_input_tokens}. Increase the limit; evaluation will not silently truncate schemas."
            )
        rendered.append((token_count, row, prompt))
    rendered.sort(key=lambda item: item[0])

    batch_size = int(args.batch_size or spec.get("batch_size", 8))
    candidate_selection = str(spec.get("candidate_selection", args.candidate_selection))
    cursor = 0
    gold_cache: dict[tuple[str, str, bool], QueryResult] = {}
    progress = tqdm(total=len(examples), initial=len(completed_ids), desc=slug, unit="example", dynamic_ncols=True)
    update_status(status_path, phase="evaluating", attention=attention, batch_size=batch_size, cuda=cuda_memory())
    try:
        while cursor < len(rendered):
            batch_items = rendered[cursor : cursor + batch_size]
            prompts = [item[2] for item in batch_items]
            try:
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=False,
                ).to(model.device)
                generation_started = time.monotonic()
                with torch.inference_mode():
                    generation_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                    if args.num_candidates > 1:
                        generation_kwargs.update(
                            {
                                "do_sample": True,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "num_return_sequences": args.num_candidates,
                            }
                        )
                    else:
                        generation_kwargs["do_sample"] = False
                    generated = model.generate(**encoded, **generation_kwargs)
                torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                if "encoded" in locals():
                    del encoded
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                logger.warning("CUDA OOM for %s; reducing batch size to %d and retrying.", slug, batch_size)
                update_status(status_path, batch_size=batch_size, oom_recovery=True, cuda=cuda_memory())
                continue

            elapsed_ms = (time.monotonic() - generation_started) * 1000
            input_width = encoded["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
            batch_predictions = []
            expected_outputs = len(batch_items) * args.num_candidates
            if len(decoded) != expected_outputs:
                raise RuntimeError(
                    f"Expected {expected_outputs} generated sequences, received {len(decoded)}"
                )
            for item_index, (input_tokens, row, _prompt) in enumerate(batch_items):
                raw_candidates = decoded[
                    item_index * args.num_candidates : (item_index + 1) * args.num_candidates
                ]
                candidate_rows = []
                candidate_results = []
                for candidate_index, candidate_raw in enumerate(raw_candidates):
                    candidate_sql = extract_sql(candidate_raw)
                    candidate_match, _candidate_gold, candidate_result = execution_match(
                        args.project_root, row, candidate_sql, args.query_timeout, gold_cache
                    )
                    candidate_results.append(candidate_result)
                    candidate_rows.append(
                        {
                            "index": candidate_index,
                            "raw_prediction": candidate_raw,
                            "predicted_sql": candidate_sql,
                            "execution_status": candidate_result.status,
                            "execution_error": candidate_result.error,
                            "execution_match": candidate_match,
                        }
                    )
                if candidate_selection == "value-aware-voting":
                    selected_index, consensus_votes = select_value_aware_voting(candidate_results)
                else:
                    selected_index, consensus_votes = select_execution_consensus(candidate_results)
                raw = raw_candidates[selected_index]
                predicted_sql = extract_sql(raw)
                gold_sql = str(row["sql"])
                predicted_canonical, syntax_error = canonical_sql(predicted_sql)
                gold_canonical, gold_syntax_error = canonical_sql(gold_sql)
                matched, gold_result, predicted_result = execution_match(
                    args.project_root, row, predicted_sql, args.query_timeout, gold_cache
                )
                stripped_raw = raw.strip()
                record = {
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "complexity": row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"),
                    "question": row["question"],
                    "gold_sql": gold_sql,
                    "raw_prediction": raw,
                    "predicted_sql": predicted_sql,
                    "raw_exact_match": stripped_raw == gold_sql.strip(),
                    "normalized_exact_match": normalize_sql(predicted_sql) == normalize_sql(gold_sql),
                    "canonical_exact_match": bool(
                        predicted_canonical is not None
                        and gold_canonical is not None
                        and predicted_canonical == gold_canonical
                    ),
                    "syntax_valid": predicted_canonical is not None,
                    "syntax_error": syntax_error,
                    "gold_syntax_error": gold_syntax_error,
                    "execution_match": matched,
                    "gold_execution_status": gold_result.status,
                    "prediction_execution_status": predicted_result.status,
                    "prediction_execution_error": predicted_result.error,
                    "format_compliant": stripped_raw == predicted_sql and bool(READ_ONLY_PREFIX.match(stripped_raw)),
                    "input_tokens": input_tokens,
                    "input_truncated": False,
                    "output_tokens": len(tokenizer(raw, add_special_tokens=False)["input_ids"]),
                    "generation_ms_per_example": round(elapsed_ms / len(batch_items), 3),
                    "num_candidates": args.num_candidates,
                    "selected_candidate_index": selected_index,
                    "execution_consensus_votes": consensus_votes,
                    "candidate_selection": candidate_selection,
                    "candidate_oracle_match": any(
                        bool(candidate["execution_match"]) for candidate in candidate_rows
                    ),
                    "candidates": candidate_rows if args.num_candidates > 1 else None,
                }
                batch_predictions.append(record)
            append_predictions(prediction_path, batch_predictions)
            completed_rows.extend(batch_predictions)
            cursor += len(batch_items)
            progress.update(len(batch_items))
            progress.set_postfix(batch=batch_size, vram=f"{cuda_memory().get('allocated_gib', 0):.1f}G")
            update_status(
                status_path,
                phase="evaluating",
                current_model=slug,
                completed_examples=len(completed_rows),
                total_examples=len(examples),
                batch_size=batch_size,
                cuda=cuda_memory(),
            )
    finally:
        progress.close()

    metrics = summarize(completed_rows)
    metrics.update(
        {
            "slug": slug,
            "tier": spec.get("tier"),
            "repo_id": spec["repo_id"],
            "revision": spec.get("revision"),
            "base_slug": spec.get("base_slug", spec["slug"]),
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "adapter_label": args.adapter_label,
            "attention": attention,
            "dtype": str(next(model.parameters()).dtype),
            "load_seconds": load_seconds,
            "evaluation_seconds": round(time.monotonic() - started - load_seconds, 3),
            "final_batch_size": batch_size,
            "num_candidates": args.num_candidates,
            "candidate_selection": "execution_consensus" if args.num_candidates > 1 else "greedy",
            "temperature": args.temperature if args.num_candidates > 1 else None,
            "top_p": args.top_p if args.num_candidates > 1 else None,
            "cuda_peak": cuda_memory(),
        }
    )
    atomic_json(output_dir / "metrics.json", metrics)
    logger.info(
        "%s complete: execution=%.3f%% syntax=%.3f%% normalized_EM=%.3f%%",
        slug,
        metrics["execution_match_pct"],
        metrics["syntax_valid_pct"],
        metrics["normalized_exact_match_pct"],
    )

    try:
        del generated
    except UnboundLocalError:
        pass
    try:
        del encoded
    except UnboundLocalError:
        pass
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    logger.info("%s unloaded; CUDA=%s", slug, cuda_memory())
    update_status(status_path, phase="model_complete", current_model=slug, cuda=cuda_memory())
    return metrics


def write_comparison(output_dir: Path, metrics: Sequence[dict[str, Any]]) -> None:
    atomic_json(output_dir / "comparison.json", list(metrics))
    columns = (
        "tier",
        "slug",
        "examples",
        "execution_match_pct",
        "syntax_valid_pct",
        "normalized_exact_match_pct",
        "canonical_exact_match_pct",
        "format_compliant_pct",
        "mean_generation_ms_per_example",
        "max_allocated_gib",
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in metrics:
            row = {column: item.get(column) for column in columns}
            row["max_allocated_gib"] = item.get("cuda_peak", {}).get("max_allocated_gib")
            writer.writerow(row)


def create_archive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if path.resolve() == destination.resolve() or path.suffix == ".tmp":
                continue
            archive.add(path, arcname=path.relative_to(source), recursive=False)


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.manifest = args.manifest.resolve()
    args.data = args.data.resolve()
    args.project_root = args.project_root.resolve()
    args.model_root = args.model_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.adapter_dir is not None:
        args.adapter_dir = args.adapter_dir.resolve()
        if not args.adapter_label:
            raise ValueError("--adapter-label is required with --adapter-dir")
        if not args.adapter_dir.is_dir():
            raise FileNotFoundError(args.adapter_dir)
    elif args.adapter_label:
        raise ValueError("--adapter-label requires --adapter-dir")
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be at least one")
    if not 0.0 < args.temperature:
        raise ValueError("--temperature must be greater than zero")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    logger = configure_logging(args.output_dir)
    status_path = args.output_dir.parent / "status.json"

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation")
    selected = set(args.models) if args.models else None
    specs = load_specs(args.config, args.manifest, selected)
    if args.adapter_dir is not None:
        if len(specs) != 1:
            raise ValueError("Adapter evaluation requires exactly one --model")
        base_slug = str(specs[0]["slug"])
        specs[0] = {**specs[0], "base_slug": base_slug, "slug": str(args.adapter_label)}
    examples = read_jsonl(args.data)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("Evaluation dataset is empty")

    torch.manual_seed(args.seed)
    gpu = torch.cuda.get_device_properties(0)
    run_config = {
        "models": specs,
        "examples": len(examples),
        "data": str(args.data),
        "model_source": args.model_source,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_candidates": args.num_candidates,
        "candidate_selection": "execution_consensus" if args.num_candidates > 1 else "greedy",
        "temperature": args.temperature if args.num_candidates > 1 else None,
        "top_p": args.top_p if args.num_candidates > 1 else None,
        "query_timeout": args.query_timeout,
        "seed": args.seed,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "adapter_label": args.adapter_label,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": gpu.name,
        "gpu_total_gib": round(gpu.total_memory / 1024**3, 3),
    }
    atomic_json(args.output_dir / "run_config.json", run_config)
    update_status(status_path, phase="starting", **run_config)
    logger.info("Starting on %s (%.2f GiB), torch=%s transformers=%s", gpu.name, run_config["gpu_total_gib"], torch.__version__, transformers.__version__)

    all_metrics = []
    try:
        for index, spec in enumerate(specs, start=1):
            update_status(status_path, model_index=index, model_count=len(specs))
            all_metrics.append(evaluate_model(spec, examples, args, logger, status_path))
            write_comparison(args.output_dir, all_metrics)
        update_status(status_path, phase="complete", current_model=None, completed_models=len(all_metrics))
        if args.archive:
            create_archive(args.output_dir, args.archive.resolve())
            logger.info("Results archive: %s", args.archive.resolve())
        print(f"TEXT2SQL_EVAL_COMPLETE={args.output_dir}")
        return 0
    except Exception as exc:
        update_status(status_path, phase="failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
