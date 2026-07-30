# Final model release

The selected model is distributed as a **base model plus PEFT/QLoRA adapter**.
Do not upload or merge the 4B base weights into the team Drive artifact.

Download the selected adapter here:

**[Shared Google Drive — Qwen3 text-to-SQL QLoRA adapter](https://drive.google.com/drive/folders/1gXXcYMg5Ejlmh-UpG9yDKNScoTieGYcD?usp=sharing)**

## Exact release identity

- Release name: `talk-to-your-database-qwen3-4b-spider-qlora-v1`
- Base: `Qwen/Qwen3-4B-Instruct-2507`
- Base revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Selected training lineage: run `20260720-213353`
- Training data: 6,997-example natural-distribution Spider package
- Adapter weights: `adapter_model.safetensors`
- Adapter size: 132,187,888 bytes
- Adapter SHA-256:
  `5274d4c15179b195443940d92f8caacf10f99bdfca106ee24324cd44a2fbe9bb`
- Inference prompt: M-Schema, three bounded example values per eligible column

The hard-example curriculum and Gretel-augmented Qwen3 adapters are not the
selected release because they scored below the natural-distribution adapter.

## What to upload to shared Drive

Upload the complete `final_adapter/` directory from selected run
`20260720-213353`, renamed to:

```text
talk-to-your-database-qwen3-4b-spider-qlora-v1/
```

It contains eleven files listed in `final_model.json`. Uploading the complete
directory preserves the training tokenizer and chat template, although the two
strictly essential PEFT files are `adapter_model.safetensors` and
`adapter_config.json`.

Do **not** upload:

- Qwen base-model `.safetensors` files;
- `optimizer.pt`, scheduler/RNG state, or full checkpoints;
- raw training data, databases, credentials, or logs containing local paths.

The shared folder should grant the six team members Viewer access. Public
access should remain enabled only if the upstream model/data licenses and the
team's release policy allow public redistribution.

## Verify before and after upload

```bash
python scripts/verify_final_adapter.py \
  --adapter-dir /path/to/talk-to-your-database-qwen3-4b-spider-qlora-v1
```

Archive for Drive only after verification:

```bash
tar -czf talk-to-your-database-qwen3-4b-spider-qlora-v1.tar.gz \
  talk-to-your-database-qwen3-4b-spider-qlora-v1

shasum -a 256 talk-to-your-database-qwen3-4b-spider-qlora-v1.tar.gz
```

Record the archive SHA-256 alongside the Drive file. After a teammate downloads
it, rerun the verifier; it validates the weight hash, byte size, PEFT settings,
base repository, and base revision.

## Load

The public base is downloaded directly from Hugging Face and the adapter is
attached with PEFT. A generation-only check is provided:

```bash
python scripts/smoke_final_model.py \
  --adapter-dir /path/to/talk-to-your-database-qwen3-4b-spider-qlora-v1 \
  --schema-file /path/to/schema.txt \
  --question "How many singers are older than 30?"
```

Use an NVIDIA GPU for this command. The full benchmark path is the adapter mode
of `scripts/run_colab_zero_shot_eval.sh`, which additionally performs SQL
parsing, read-only execution, and scoring.
