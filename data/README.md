# Data reconstruction

Only lightweight provenance, EDA, manifests, and checksums are committed.
Generated examples and SQLite databases are deliberately excluded.

## Required upstream payload

Download the official Spider 1.0 database payload from the official Spider
project and extract it as:

```text
milestone3/database/
├── concert_singer/concert_singer.sqlite
├── ...
└── <database_id>/<database_id>.sqlite
```

Do not flatten the database directories. The scripts resolve database paths
relative to the repository root.

## Build the canonical split

```bash
bash scripts/setup_spider_data.sh
```

This creates a dedicated `.venv-data`, downloads the pinned annotation Parquet
files, derives schema metadata from SQLite, executes gold SQL with read-only
guards, and creates:

- 6,997 accepted training examples;
- 1,034 official validation examples;
- 166 training databases and 20 validation databases;
- zero train/validation database overlap;
- three rejected non-executable training annotations;
- schema, provenance, checksums, and EDA.

Reference metadata lives in `data/processed/spider/`. Run output hashes should
match its manifest when the same upstream database payload and annotation
revision are used.

## Build training variants

Natural-distribution and hard-example curriculum:

```bash
python scripts/build_sft_dataset.py
python scripts/preflight_sft_dataset.py
```

Selected inference-format M-Schema package:

```bash
python scripts/build_mschema_sft_package.py
```

Optional rejected/diagnostic variants:

```bash
python scripts/build_gretel_augmented_sft.py \
  --parquet data/raw/gretel/gretel-synthetic-text2sql-train.parquet

python scripts/build_gradesql_orm_sft.py \
  --input data/raw/gradesql/spider-balanced.parquet
```

The Gretel and GradeSQL sources are not required to reproduce the selected
Qwen3 model. They are retained as negative/diagnostic experiment pipelines.

## Leakage and safety contract

- Keep the official Spider train and validation splits unchanged.
- Never randomly repartition examples across databases.
- Assistant targets contain SQL only.
- Training gold queries must execute under immutable read-only SQLite access.
- Validation examples must never enter SFT or verifier training.
- Selection claims are validation-set claims until confirmed on a separate
  untouched test set.

