import sqlite3
import pandas as pd
import os
from tqdm import tqdm

SPIDER_DB = r"spider\database"      # Use a raw string
os.makedirs("output", exist_ok=True)

summary = []
columns = []

for db in tqdm(os.listdir(SPIDER_DB)):

    db_path = os.path.join(SPIDER_DB, db, f"{db}.sqlite")

    if not os.path.exists(db_path):
        continue

    conn = sqlite3.connect(db_path)

    # Ignore invalid UTF-8 characters
    conn.text_factory = lambda b: b.decode(errors="ignore")

    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';",
            conn
        )
    except Exception as e:
        print(f"Skipping database {db}: {e}")
        conn.close()
        continue

    for table in tables["name"]:

        try:
            df = pd.read_sql(
                f"SELECT * FROM '{table}'",
                conn
            )
        except Exception as e:
            print(f"Skipping {db}.{table}: {e}")
            continue

        summary.append({
            "Database": db,
            "Table": table,
            "Rows": len(df),
            "Columns": len(df.columns)
        })

        for c in df.columns:
            columns.append({
                "Database": db,
                "Table": table,
                "Column": c,
                "Datatype": str(df[c].dtype),
                "Missing": df[c].isna().sum(),
                "Distinct": df[c].nunique(dropna=True)
            })

    conn.close()

summary = pd.DataFrame(summary)
columns = pd.DataFrame(columns)

summary.to_csv(
    "output/database_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

columns.to_csv(
    "output/column_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("EDA Completed")