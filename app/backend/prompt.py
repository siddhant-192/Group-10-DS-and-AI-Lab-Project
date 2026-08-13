"""M-Schema chat prompt wrapper (same text as smoke_final_model.mschema_prompt)."""

from __future__ import annotations


def mschema_prompt(schema: str, question: str) -> str:
    return (
        "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
        f"【Schema】\n{schema}\n\n"
        f"【Question】\n{question}\n\n"
        "【Evidence】\n\n"
        "Please read and understand the database schema carefully, and generate an executable SQL "
        "query based on the user's question and evidence. "
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
