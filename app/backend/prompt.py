"""M-Schema chat prompt wrapper (same text as smoke_final_model.mschema_prompt)."""

from __future__ import annotations

# Real rules live under "sqlite". Other keys are extension placeholders only.
DIALECT_INSTRUCTIONS = {
    "sqlite": (
        "Generate SQLite SQL only. Do not use functions or syntax from other engines. "
        "For dates use SQLite strftime / date / datetime, for example "
        "CAST(strftime('%Y', col) AS INTEGER) to get a year. "
        "Use || for string concatenation, LIMIT for row caps, LENGTH and SUBSTR "
        "for strings. Do not emit vendor helpers that SQLite does not implement."
    ),
    # Placeholders — fill in if a non-SQLite backend is added later.
    "mysql": None,
    "postgres": None,
}


def _dialect_rules(dialect: str) -> str:
    key = (dialect or "sqlite").strip().lower() or "sqlite"
    rules = DIALECT_INSTRUCTIONS.get(key)
    if rules:
        return rules
    return (
        f"Generate valid {key} SQL only. "
        "Detailed dialect rules for this engine are not configured yet "
        f"(placeholder). Until they are added, prefer portable SQL."
    )


def mschema_prompt(schema: str, question: str, dialect: str = "sqlite") -> str:
    dialect = (dialect or "sqlite").strip().lower() or "sqlite"
    dialect_rules = _dialect_rules(dialect)
    return (
        f"You are now a {dialect} data analyst, and you are given a database schema as follows:\n\n"
        f"【Schema】\n{schema}\n\n"
        f"【Question】\n{question}\n\n"
        "【Evidence】\n\n"
        "Please read and understand the database schema carefully, and generate an executable SQL "
        "query based on the user's question and evidence. "
        f"{dialect_rules} "
        "Use ONLY tables and columns explicitly present in the supplied schema. "
        "Never invent or assume a table, column, metric, relationship, or derived field that "
        "is not supported by the schema. "
        "If a requested concept is not represented directly in the schema, derive it only "
        "when it can be computed from existing schema elements. "
        "For ambiguous ranking words such as 'top', 'best', or 'popular', do not invent "
        "a ranking column such as TotalSales. Use a ranking criterion only when it is "
        "supported by the schema and the question/context; otherwise produce no fabricated "
        "interpretation. "
        "Output the SQL only, protected by ```sql and ```. "
        "Do not write a long explanation before the SQL fence."
    )
