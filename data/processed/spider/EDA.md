# Spider Data Preparation Report

Generated: `2026-07-20T07:35:59.725949+00:00`

## Readiness verdict

The processed data is structurally ready for a first supervised fine-tuning experiment. Non-executable training annotations were excluded from `train.jsonl`; preserve the official validation split.

## Database inventory

- Databases: **166**
- SQLite quick checks passing: **166**
- Tables: **873**
- Columns: **4497**
- Foreign keys: **800**

## Official splits

| Split | Official examples | Usable JSONL | Databases | Unique SQL | Execution status |
|---|---:|---:|---:|---:|---|
| train | 7000 | 6997 | 140 | 3960 | error=3, ok=6997 |
| validation | 1034 | 1034 | 20 | 563 | ok=1034 |

## Rejected source annotations

- Training: **3**
- Validation: **0**
- Full SQL, questions, database IDs, and SQLite errors are preserved in `validation_failures.jsonl`.

## Leakage checks

- Train/validation database overlap: **0**
- Normalized question overlap: **6**
- Exact `(db_id, question, SQL)` overlap: **0**
- Local databases unused by labeled splits: `academic, geo, imdb, restaurants, scholar, yelp`

## Query structure

The following labels are transparent structural proxies, not Spider's official hardness labels.

| Split | Simple | Moderate | Complex | Join | Subquery | Set operation |
|---|---:|---:|---:|---:|---:|---:|
| train | 3995 | 2243 | 762 | 2771 | 1019 | 526 |
| validation | 593 | 336 | 105 | 408 | 159 | 80 |

## Training contract

Each JSONL row contains `messages` in chat fine-tuning format plus the raw `question`, `sql`, serialized `schema`, database ID, structural features, and execution-validation result. The assistant target is SQL only.

Do not merge or randomly reshuffle the official splits: validation databases are deliberately unseen during training.
