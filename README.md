# Talk to Your Database — Milestone 3 Reproducibility Package

This repository contains the code, configurations, and compact evidence needed
to reproduce the model experiments for **Talk to Your Database**, a
natural-language to SQL system for small open-source language models. The
Milestone 3 report is maintained and submitted separately.

The selected project model is **Qwen/Qwen3-4B-Instruct-2507** with our
natural-distribution Spider QLoRA adapter. The base checkpoint remains on
Hugging Face; only the approximately 132 MB adapter must be distributed
separately. With an M-Schema prompt, the selected model achieved:

- **78.627%** strict execution accuracy (813/1,034);
- **83.172%** MAC-SQL/FINER-compatible execution accuracy (860/1,034);
- **100%** syntactically valid SQL on the Spider validation split.

## What is included

```text
.
├── configs/          Model, sampling, and QLoRA policies
├── data/             EDA, provenance, checksums, and rebuild instructions
├── docs/             Experiment ledger and workflow documentation
├── evidence/         Compact metrics and comparison tables (no predictions)
├── figures/          Reusable Mermaid architecture diagrams
├── models/           Pinned Hugging Face download manifest (no weights)
├── release/          Exact final-adapter identity and distribution guide
├── scripts/          Data, evaluation, training, ensemble, and audit tools
├── src/              Reusable Spider data-pipeline modules
└── tests/            Unit tests for data, evaluation, and training contracts
```

Large or sensitive runtime material is deliberately absent: model weights,
QLoRA binaries, SQLite databases, generated JSONL datasets, predictions,
checkpoints, virtual environments, Colab credentials, and rclone
configuration. See [`PACKAGE_CONTENTS.md`](PACKAGE_CONTENTS.md) for the complete
inclusion policy.

## Reproduction map

| Goal | Command or document |
|---|---|
| Validate the repository | `bash scripts/audit_public_package.sh` |
| Run unit tests | `python -m pytest -q` |
| Build Spider data | `bash scripts/setup_spider_data.sh` |
| Build chat-SFT data | `python scripts/build_sft_dataset.py` |
| Build M-Schema prompts | `python scripts/build_mschema_sft_package.py` |
| Pin/download baseline models | `bash scripts/download_eval_models.sh` |
| Run 12-row baseline smoke test | `bash scripts/run_colab_zero_shot_eval.sh --limit 12` |
| Run the full three-model baseline | `bash scripts/run_colab_zero_shot_eval.sh` |
| Run QLoRA resume smoke test | `bash scripts/run_colab_qlora_sft.sh --smoke` |
| Train the selected configuration | `bash scripts/run_colab_qlora_sft.sh --model qwen3-4b-instruct-2507 --dataset base` |
| Verify a downloaded final adapter | `python scripts/verify_final_adapter.py --adapter-dir PATH` |
| Run a generation smoke test | `python scripts/smoke_final_model.py --adapter-dir PATH --schema-file SCHEMA --question "..."` |

## 1. Local setup

Python 3.10 or newer is required. Python 3.12 was used for the local pipeline.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Model training is not intended for a laptop. The local environment prepares
data, builds bundles, launches Colab work, downloads results, and performs
offline evaluation.

## 2. Acquire the Spider databases and rebuild data

The annotation downloader is automated, but the official Spider 1.0 SQLite
database payload is not redistributed here. Obtain it from the official Spider
project under its license and extract it so the layout is:

```text
milestone3/database/<database_id>/<database_id>.sqlite
```

Then run:

```bash
bash scripts/setup_spider_data.sh
python scripts/build_sft_dataset.py
python scripts/build_mschema_sft_package.py
```

The expected outputs and reference hashes are documented in
[`data/README.md`](data/README.md). Generated JSONL, Parquet, and SQLite files
are gitignored.

## 3. Set up terminal-controlled Google Colab

```bash
bash scripts/setup_colab_cli.sh
```

Follow the Google authentication prompt shown by the script. Authentication is
stored outside this repository. The setup performs an upload, execution,
download, and cleanup smoke test. The launchers request an NVIDIA L4, stream
logs and progress, collect results, and stop the runtime automatically.

Run the baseline pilot:

```bash
bash scripts/run_colab_zero_shot_eval.sh --limit 12
```

Run the full fixed baseline:

```bash
bash scripts/run_colab_zero_shot_eval.sh
```

The three base checkpoints are loaded sequentially so their weights do not
coexist in VRAM. Exact repository revisions and file hashes are recorded in
[`models/text2sql-eval/download_manifest.json`](models/text2sql-eval/download_manifest.json).
All three repositories were public at experiment time; a Hugging Face token is
not embedded or required under normal anonymous rate limits.

## 4. Reproduce QLoRA training

The selected experiment uses one epoch over 6,997 execution-validated Spider
training examples:

```bash
bash scripts/run_colab_qlora_sft.sh --smoke

bash scripts/run_colab_qlora_sft.sh \
  --model qwen3-4b-instruct-2507 \
  --dataset base
```

The policy in [`configs/text2sql_qlora_training.json`](configs/text2sql_qlora_training.json)
uses 4-bit NF4 QLoRA, rank 16, alpha 32, dropout 0.05, all linear projection
modules, learning rate `2e-4`, cosine scheduling, bfloat16, gradient
checkpointing, and assistant-only loss. The full selected run completed 438
optimizer steps with validation loss 0.255960.

The training launcher exports resumable checkpoints while training, validates
their hashes and members, downloads final artifacts, and terminates the Colab
runtime. See [`docs/qlora-training.md`](docs/qlora-training.md) for recovery and
monitoring instructions.

## 5. Evaluate the tuned model

Download the
[final Qwen3 QLoRA adapter from the shared Google Drive folder](https://drive.google.com/drive/folders/1gXXcYMg5Ejlmh-UpG9yDKNScoTieGYcD?usp=sharing),
place the extracted adapter directory anywhere outside Git, verify it, and
evaluate it against the same 1,034 examples:

```bash
python scripts/verify_final_adapter.py \
  --adapter-dir /path/to/talk-to-your-database-qwen3-4b-spider-qlora-v1

bash scripts/run_colab_zero_shot_eval.sh \
  --model qwen3-4b-instruct-2507 \
  --adapter-dir /path/to/talk-to-your-database-qwen3-4b-spider-qlora-v1 \
  --adapter-label qwen3-base-sft
```

To recreate the selected M-Schema inference input first:

```bash
python scripts/build_xiyan_mschema_eval_data.py
```

The exact base revision, adapter hash, file list, and selected metrics are in
[`release/final_model.json`](release/final_model.json). The base model and
adapter are loaded separately through PEFT; no merged 4B checkpoint needs to be
shared.

## Results snapshot

| System | Strict execution | Compatible execution | Decision |
|---|---:|---:|---|
| Qwen3-4B zero-shot | 72.340% | — | Strongest original baseline |
| Qwen2.5-Coder-1.5B zero-shot | 56.576% | — | Efficiency baseline |
| DeepSeek-Coder-1.3B zero-shot | 47.292% | — | Older/weak baseline |
| Qwen3-4B natural QLoRA + DDL | 76.886% | — | Selected adapter |
| **Qwen3-4B natural QLoRA + M-Schema** | **78.627%** | **83.172%** | **Primary architecture** |
| XiYanSQL-3B + M-Schema | 78.433% | 83.269% | External specialist comparison |
| FINER-SQL-3B, 30 candidates | 79.014% | 84.236% | Slow sampling comparison |
| Five-model execution consensus | 82.785% | 87.331% | Optional high-accuracy mode |
| Strict FINER fallback | 83.075% | 87.331% | Best measured deployable selector |

These figures are empirical Spider validation results, not claims about the
hidden Spider test set. Full compact evidence is indexed in
[`evidence/README.md`](evidence/README.md).

## Safety and reproducibility

- SQL is restricted to one `SELECT` or `WITH` statement.
- SQLite is opened immutable/read-only and write actions are denied.
- Queries have execution time and row limits.
- Train and validation databases remain disjoint.
- Model revisions, dataset outputs, adapter weights, and artifacts are hashed.
- No credentials are accepted through committed configuration.
- Colab sessions are stopped after result collection, including most failures.

Run the repository audit before every public push:

```bash
bash scripts/audit_public_package.sh
```

## Licensing

This package preserves upstream attribution, but no license for the team's
original code has been selected in this snapshot. The repository owner should
choose and add a project license before treating the repository as generally
reusable. Spider-derived material remains subject to Spider's CC BY-SA 4.0
terms; model use remains subject to each Hugging Face model's license. See
[`THIRD_PARTY.md`](THIRD_PARTY.md).
