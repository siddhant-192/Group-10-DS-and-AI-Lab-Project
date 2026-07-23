# Package contents and exclusion policy

This directory is a curated Milestone 3 publication package, not a byte-for-byte
copy of the development workspace.

## Included

- Python and shell source for data preparation, model downloading, Colab
  orchestration, evaluation, QLoRA training, prompt variants, ensembles, and
  verifier experiments;
- model, sampling, and training configuration files;
- unit tests;
- Spider EDA, source provenance, output hashes, and SFT-package manifests;
- exact public model revisions and downloaded-file hashes;
- compact aggregate metrics and error-analysis tables;
- final adapter identity, expected file inventory, and verification code;
- architecture and sequence diagrams.

## Excluded from Git

| Material | Reason | How to reproduce or distribute |
|---|---|---|
| Base-model weights | Approximately 13 GiB across the baseline set | Download from Hugging Face using the pinned manifest |
| Final QLoRA adapter binary | 132,187,888-byte file exceeds GitHub's normal 100 MB object limit | Share the adapter directory through Drive or a model registry |
| Training checkpoints and optimizer state | Multi-gigabyte, transient, and not needed for inference | Recreate with the QLoRA launcher; keep in private cold storage |
| Spider SQLite payload | Approximately 933 MiB and controlled by upstream data terms | Obtain from the official Spider project |
| Generated train/validation JSONL | Regenerable and up to hundreds of MiB across variants | Run the checked-in data builders |
| Raw Parquet datasets | Regenerable and subject to upstream terms | Use the recorded sources/builders |
| Model predictions | Large experiment outputs that can be regenerated | Run the pinned evaluator; aggregate metrics are retained |
| Virtual environments/caches | Platform-specific and regenerable | Install from requirements files |
| Colab/Google/rclone credentials | Sensitive and machine-specific | Authenticate locally; never commit credentials |
| Internal Drive storage ledger | Contains workstation-specific storage routing | Maintained outside the public package |

All copied machine paths were converted to project-relative paths. A
high-confidence credential scan, private-path scan, large-file scan, shell
syntax check, JSON parse check, Python compilation, and unit tests are run
before handoff.
