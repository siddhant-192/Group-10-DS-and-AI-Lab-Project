"""XiYan-style M-Schema renderer (lifted from build_xiyan_mschema_eval_data.py)."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3


EMAIL = re.compile(r"^[\w.-]+@[\w.-]+\.\w+$")


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def examples_for(
    connection: sqlite3.Connection, table: str, column: str, limit: int = 5
) -> list[str]:
    try:
        rows = connection.execute(
            f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column)} IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    values = [str(row[0]) for row in rows if row[0] is not None and str(row[0])]
    if any(EMAIL.match(value) or "http://" in value or "https://" in value for value in values):
        return []
    return values


def render_mschema(database: Path, db_id: str, example_num: int = 3) -> str:
    """Build M-Schema text for prompts from a SQLite file."""

    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        output = [f"【DB_ID】 {db_id}", "【Schema】"]
        foreign_keys: list[str] = []
        for table in tables:
            output.append(f"# Table: {table}")
            field_lines: list[str] = []
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
            for foreign in connection.execute(
                f"PRAGMA foreign_key_list({quote_identifier(table)})"
            ):
                _id, _seq, referred_table, constrained, referred, *_rest = foreign
                foreign_keys.append(f"{table}.{constrained}={referred_table}.{referred}")
        if foreign_keys:
            output.append("【Foreign keys】")
            output.extend(foreign_keys)
        return "\n".join(output)
    finally:
        connection.close()
