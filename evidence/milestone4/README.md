# Milestone 4 compact evidence

This directory contains the small, reviewable artifacts behind the training,
hyperparameter, and checkpoint claims in the Milestone 4 PDF. It excludes
adapter weights, optimizer checkpoints, predictions, Colab bundles, and logs.

## Contents

- `hparam/` contains the 13-trial aggregate tables, compatible-execution
  metrics, and the frozen selection record.
- `decoding/` contains the inference-only sampling/selection study.
- `final_training/` contains the trainer loss history, final loss, environment,
  tokenization summary, run manifest, and adapter checksum verification.
- `evaluation/` contains strict and compatible execution metrics for the final
  full-data adapter on the 1,001-example internal tuning split.

## Key results

| Artifact | Strict EX | Compatible EX | Syntax |
| --- | ---: | ---: | ---: |
| Best 2,048-row screening adapter | 86.513% | 89.111% | 100.0% |
| Final 5,996-row checkpoint 375 | 85.514% | 89.011% | 100.0% |

The final adapter is identified by SHA-256
`63a51ff491a163c1433dd4ac56d936d969337de67279f852ea1ef966ac335e5c`.
Its binary is intentionally not committed because it is 132,187,888 bytes.

Regenerate the figures with:

```bash
python src/scripts/plot_milestone4_evidence.py
```
