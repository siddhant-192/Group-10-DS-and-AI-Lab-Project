# Demo SQLite databases

Place `.sqlite` files here. The UI registry uses each file's stem as `db_id`.

Examples:

- `chinook.sqlite`
- `concert_singer.sqlite` (copy from Spider after download)

Populate with:

```bash
python app/scripts/download_demo_databases.py
```

Optional: copy Spider DBs if you already have them:

```bash
python app/scripts/download_demo_databases.py --copy-spider-from path/to/milestone3/database
```

`.sqlite` files are gitignored; keep this folder for local/Colab demos only.
