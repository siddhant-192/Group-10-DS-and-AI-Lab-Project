"""M-Schema chat prompt wrapper (same text as smoke_final_model.mschema_prompt)."""

from __future__ import annotations


def mschema_prompt(schema: str, question: str) -> str:
    return (
        "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
        f"【Schema】\n{schema}\n\n"
        f"【Question】\n{question}\n\n"
        "【Evidence】\n\n"
        "Please read and understand the database schema carefully, and generate an executable SQL based "
        "on the user's question and evidence. "
        "Output the SQL only, protected by ```sql and ```. "
        "Do not write a long explanation before the SQL fence."
    )
