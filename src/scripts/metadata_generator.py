import os
import json
import sqlite3
from tqdm import tqdm

SPIDER_DB = "spider/database"
OUTPUT = "output"

os.makedirs(OUTPUT, exist_ok=True)

schema = {}

for db in tqdm(os.listdir(SPIDER_DB)):

    db_path = os.path.join(
        SPIDER_DB,
        db,
        f"{db}.sqlite"
    )

    if not os.path.exists(db_path):
        continue

    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    schema[db] = {}

    for table in tables:

        table = table[0]

        info = cursor.execute(
            f"PRAGMA table_info('{table}')"
        ).fetchall()

        schema[db][table] = []

        for col in info:

            schema[db][table].append({

                "column": col[1],

                "datatype": col[2],

                "pk": bool(col[5])

            })

    conn.close()

with open("output/schema.json", "w") as f:

    json.dump(schema, f, indent=4)

print("Schema Extraction Complete")