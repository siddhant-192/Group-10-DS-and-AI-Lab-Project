"""Deterministic SQLite schema introspection and prompt serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
from typing import Any


def quote_identifier(identifier: str) -> str:
    """Quote an SQLite identifier without changing its spelling."""

    return '"' + identifier.replace('"', '""') + '"'


def readonly_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro&immutable=1"


@dataclass(frozen=True)
class Column:
    cid: int
    name: str
    declared_type: str
    not_null: bool
    default_value: Any
    primary_key_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForeignKey:
    id: int
    sequence: int
    from_column: str
    referenced_table: str
    referenced_column: str | None
    on_update: str
    on_delete: str
    match: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [column.to_dict() for column in self.columns],
            "foreign_keys": [foreign_key.to_dict() for foreign_key in self.foreign_keys],
        }


@dataclass(frozen=True)
class DatabaseSchema:
    db_id: str
    sqlite_path: str
    quick_check: str
    tables: tuple[Table, ...]

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def column_count(self) -> int:
        return sum(len(table.columns) for table in self.tables)

    @property
    def foreign_key_count(self) -> int:
        return sum(len(table.foreign_keys) for table in self.tables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_id": self.db_id,
            "sqlite_path": self.sqlite_path,
            "quick_check": self.quick_check,
            "table_count": self.table_count,
            "column_count": self.column_count,
            "foreign_key_count": self.foreign_key_count,
            "tables": [table.to_dict() for table in self.tables],
            "ddl": self.to_ddl(),
        }

    def to_ddl(self) -> str:
        """Render a compact, faithful DDL representation for model prompts."""

        statements: list[str] = []
        for table in self.tables:
            definitions: list[str] = []
            primary_key_columns = sorted(
                (column for column in table.columns if column.primary_key_position),
                key=lambda column: column.primary_key_position,
            )
            has_composite_primary_key = len(primary_key_columns) > 1

            for column in table.columns:
                declared_type = column.declared_type.strip() or "ANY"
                parts = [quote_identifier(column.name), declared_type]
                if column.not_null:
                    parts.append("NOT NULL")
                if column.primary_key_position and not has_composite_primary_key:
                    parts.append("PRIMARY KEY")
                definitions.append(" ".join(parts))

            if has_composite_primary_key:
                names = ", ".join(
                    quote_identifier(column.name) for column in primary_key_columns
                )
                definitions.append(f"PRIMARY KEY ({names})")

            for foreign_key in table.foreign_keys:
                referenced_column = (
                    quote_identifier(foreign_key.referenced_column)
                    if foreign_key.referenced_column
                    else "<implicit-primary-key>"
                )
                definitions.append(
                    "FOREIGN KEY "
                    f"({quote_identifier(foreign_key.from_column)}) REFERENCES "
                    f"{quote_identifier(foreign_key.referenced_table)} "
                    f"({referenced_column})"
                )

            body = ",\n  ".join(definitions)
            statements.append(
                f"CREATE TABLE {quote_identifier(table.name)} (\n  {body}\n);"
            )

        return "\n\n".join(statements)


def inspect_database(path: Path, relative_to: Path | None = None) -> DatabaseSchema:
    """Inspect one SQLite file in immutable read-only mode."""

    if not path.is_file():
        raise FileNotFoundError(path)

    connection = sqlite3.connect(readonly_uri(path), uri=True)
    try:
        quick_check_row = connection.execute("PRAGMA quick_check").fetchone()
        quick_check = str(quick_check_row[0]) if quick_check_row else "no-result"
        table_names = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name COLLATE NOCASE"
            )
        ]

        tables: list[Table] = []
        for table_name in table_names:
            quoted_table = quote_identifier(table_name)
            columns = tuple(
                Column(
                    cid=int(row[0]),
                    name=str(row[1]),
                    declared_type=str(row[2] or ""),
                    not_null=bool(row[3]),
                    default_value=row[4],
                    primary_key_position=int(row[5]),
                )
                for row in connection.execute(f"PRAGMA table_info({quoted_table})")
            )
            foreign_keys = tuple(
                ForeignKey(
                    id=int(row[0]),
                    sequence=int(row[1]),
                    referenced_table=str(row[2]),
                    from_column=str(row[3]),
                    referenced_column=str(row[4]) if row[4] is not None else None,
                    on_update=str(row[5]),
                    on_delete=str(row[6]),
                    match=str(row[7]),
                )
                for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})")
            )
            tables.append(Table(table_name, columns, foreign_keys))
    finally:
        connection.close()

    if relative_to is not None:
        try:
            rendered_path = str(path.resolve().relative_to(relative_to.resolve()))
        except ValueError:
            rendered_path = str(path.resolve())
    else:
        rendered_path = str(path.resolve())

    return DatabaseSchema(
        db_id=path.stem,
        sqlite_path=rendered_path,
        quick_check=quick_check,
        tables=tuple(tables),
    )
