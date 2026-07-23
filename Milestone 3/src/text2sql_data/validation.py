"""Safe, execution-aware validation for gold SQLite queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from .schema import readonly_uri


READ_ONLY_PREFIX = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)


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
class QueryValidation:
    status: str
    elapsed_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def open_readonly_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(readonly_uri(path), uri=True)


def validate_readonly_query(
    sqlite_path: Path,
    query: str,
    timeout_seconds: float = 2.0,
    connection: sqlite3.Connection | None = None,
) -> QueryValidation:
    """Execute enough of a SELECT to validate it, denying every write action."""

    started = time.monotonic()
    sql = query.strip()
    if not READ_ONLY_PREFIX.match(sql):
        return QueryValidation(
            status="unsafe",
            elapsed_ms=0.0,
            error="Gold query does not begin with SELECT or WITH.",
        )

    owns_connection = connection is None
    active_connection = connection or open_readonly_database(sqlite_path)
    deadline = started + timeout_seconds

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

    active_connection.set_authorizer(authorizer)
    active_connection.set_progress_handler(progress, 10_000)
    try:
        cursor = active_connection.execute(sql)
        cursor.fetchone()
        status = "ok"
        error = None
    except sqlite3.DatabaseError as exc:
        message = str(exc)
        status = "timeout" if "interrupted" in message.lower() else "error"
        error = message
    finally:
        active_connection.set_progress_handler(None, 0)
        active_connection.set_authorizer(None)
        if owns_connection:
            active_connection.close()

    return QueryValidation(
        status=status,
        elapsed_ms=round((time.monotonic() - started) * 1000, 3),
        error=error,
    )


def normalize_sql(query: str) -> str:
    return " ".join(query.strip().rstrip(";").split())


def query_features(query: str) -> dict[str, int | bool | str]:
    """Return transparent structural proxies, not Spider's official hardness."""

    normalized = normalize_sql(query)
    upper = normalized.upper()

    select_count = len(re.findall(r"\bSELECT\b", upper))
    join_count = len(re.findall(r"\bJOIN\b", upper))
    set_operation_count = len(re.findall(r"\b(?:UNION|INTERSECT|EXCEPT)\b", upper))
    aggregate_count = len(
        re.findall(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", upper)
    )
    has_group_by = bool(re.search(r"\bGROUP\s+BY\b", upper))
    has_order_by = bool(re.search(r"\bORDER\s+BY\b", upper))
    has_having = bool(re.search(r"\bHAVING\b", upper))
    has_where = bool(re.search(r"\bWHERE\b", upper))
    has_limit = bool(re.search(r"\bLIMIT\b", upper))
    has_distinct = bool(re.search(r"\bDISTINCT\b", upper))
    has_subquery = select_count > 1

    score = (
        join_count
        + 2 * int(has_subquery)
        + 2 * set_operation_count
        + int(has_group_by)
        + int(has_having)
        + int(has_order_by)
        + int(aggregate_count > 1)
    )
    if score <= 1:
        complexity_proxy = "simple"
    elif score <= 3:
        complexity_proxy = "moderate"
    else:
        complexity_proxy = "complex"

    return {
        "token_count": len(normalized.split()),
        "select_count": select_count,
        "join_count": join_count,
        "set_operation_count": set_operation_count,
        "aggregate_count": aggregate_count,
        "has_subquery": has_subquery,
        "has_group_by": has_group_by,
        "has_order_by": has_order_by,
        "has_having": has_having,
        "has_where": has_where,
        "has_limit": has_limit,
        "has_distinct": has_distinct,
        "complexity_proxy": complexity_proxy,
    }
