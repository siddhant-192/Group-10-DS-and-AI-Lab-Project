"""SQL extract + readonly execute for UI display (lifted/adapted from evaluate_text2sql_models)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
from urllib.parse import quote


READ_ONLY_PREFIX = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
# Prefer real SQL starts — do NOT match English "with the highest/most..."
SELECT_START = re.compile(r"\bSELECT\b", re.IGNORECASE)
WITH_CTE_START = re.compile(r"\bWITH\s+[A-Za-z_][\w]*\s+AS\s*\(", re.IGNORECASE)
FENCED_SQL = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

DEFAULT_MAX_RESULT_ROWS = None


def _sqlite_actions(*names: str) -> frozenset[int]:
    return frozenset(
        int(value)
        for name in names
        if (value := getattr(sqlite3, name, None)) is not None
    )


DENIED_ACTIONS = _sqlite_actions(
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
class ExecuteResult:
    status: str
    columns: list[str] | None
    rows: list[list[Any]] | None
    elapsed_ms: float
    error: str | None = None


def extract_sql(raw: str) -> str:
    """Pull the first SQL statement out of model text (fenced or bare).

    Avoid treating English phrases like \"with the most albums\" as SQL.
    """

    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    fenced = FENCED_SQL.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        cte = WITH_CTE_START.search(text)
        select = SELECT_START.search(text)
        starts = [m.start() for m in (cte, select) if m]
        if not starts:
            return ""
        text = text[min(starts) :]
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    text = text.strip()
    # Reject obvious non-SQL leftovers
    if not READ_ONLY_PREFIX.match(text):
        return ""
    return text


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


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, bytes):
        return value.hex()
    return value


def execute_query(
    path: Path,
    sql: str,
    timeout_seconds: float = 5.0,
    max_result_rows: int = DEFAULT_MAX_RESULT_ROWS,
) -> ExecuteResult:
    """Run a SELECT/WITH query and return columns + rows for the UI."""

    started = time.monotonic()
    if not READ_ONLY_PREFIX.match(sql):
        return ExecuteResult(
            status="unsafe",
            columns=None,
            rows=None,
            elapsed_ms=0.0,
            error="query does not begin with SELECT or WITH",
        )
    try:
        with readonly_connection(path, timeout_seconds) as connection:
            cursor = connection.execute(sql)
            columns = [str(col[0]) for col in (cursor.description or ())]
            fetched = cursor.fetchmany(max_result_rows + 1)
            if len(fetched) > max_result_rows:
                return ExecuteResult(
                    status="too_many_rows",
                    columns=columns,
                    rows=None,
                    elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                    error=f"result exceeded {max_result_rows} rows",
                )
            rows = [[_cell(cell) for cell in row] for row in fetched]
            return ExecuteResult(
                status="ok",
                columns=columns,
                rows=rows,
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                error=None,
            )
    except sqlite3.DatabaseError as exc:
        message = str(exc)
        status = "timeout" if "interrupted" in message.lower() else "error"
        return ExecuteResult(
            status=status,
            columns=None,
            rows=None,
            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            error=message,
        )
