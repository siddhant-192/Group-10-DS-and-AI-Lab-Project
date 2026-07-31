# Qwen3 database-disjoint HPO split

This directory records the reproducible metadata for the Milestone 4 split.
Generated JSONL is excluded from Git because it is large and reproducible.

The split contains:

- 5,996 training examples from 120 Spider training databases;
- 1,001 tuning examples from 20 different Spider training databases;
- zero database/schema overlap between the two partitions.

Rebuild it after generating `data/finetuning/spider_mschema_sft_v1/train_base.jsonl`:

```bash
python src/scripts/build_qwen3_hparam_split.py
python src/scripts/build_qwen3_hparam_screen.py
```

Use `checksums.json` to verify the generated `train_base.jsonl`,
`train_curriculum.jsonl`, and `validation.jsonl`. `manifest.json` records the
database IDs, complexity counts, seed, and split objective.
