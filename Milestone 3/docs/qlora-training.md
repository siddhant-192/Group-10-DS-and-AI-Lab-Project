# Text-to-SQL QLoRA training

This workflow trains one pinned model at a time on a Colab NVIDIA L4. It uses
4-bit NF4 weights, bfloat16 compute, nested quantization, gradient checkpointing,
and LoRA adapters on all linear layers. Model weights are fetched directly by
the runtime at the revision recorded in the local model manifest.

## Loss contract

The trainer renders each model's own chat template twice: once through the
assistant generation prefix and once with the gold assistant response. It then
checks that the tokenized prefix is identical and writes explicit `labels`:

- every system, schema, question, and assistant-prefix token is `-100`;
- only the SQL assistant response and its closing chat tokens receive loss;
- sequences exceeding the configured model limit raise an error;
- schema or target truncation is never permitted.

This explicit masking works consistently even when a chat template does not
provide an assistant-token mask.

## Pinned environment

The Colab requirements are in `scripts/colab-sft-requirements.txt`. The main
training policy is in `configs/text2sql_qlora_training.json`. Runtime package
versions, GPU details, model revision, data hashes, selected ID hashes, trainable
parameter count, and memory peaks are saved into each run manifest.

## Smoke test

```bash
bash scripts/run_colab_qlora_sft.sh --smoke
```

The Qwen3 curriculum smoke test selects 64 examples distributed from the
shortest through longest conversations plus 16 similarly stratified validation
examples. It trains to step 2, saves a complete checkpoint, starts a fresh model
process, restores that checkpoint, and continues to step 4. This verifies the
optimizer/scheduler state and adapter resume path rather than only adapter save.

Monitor in another terminal using the run directory printed by the launcher:

```bash
bash scripts/monitor_colab_sft.sh artifacts/qlora-training/runs/YYYYMMDD-HHMMSS
```

The launcher downloads the final adapter, metrics, logs, TensorBoard events,
trainer state, tokenization audit, run manifest, and newest resumable checkpoint.
It stops the L4 after collection, including on most failures and interrupts.

Full runs also protect against abrupt Colab VM deletion. Every complete training
checkpoint is atomically archived on the remote runtime, downloaded while
training continues, and verified locally for size, SHA-256, safe paths, adapter
weights, optimizer, scheduler, RNG, and trainer state. Verified exports are kept
under the run's `checkpoint-exports/` directory.

If Colab deletes a runtime before final collection, resume on a fresh L4 with:

```bash
bash scripts/run_colab_qlora_sft.sh \
  --model qwen3-4b-instruct-2507 --dataset base \
  --resume-checkpoint artifacts/qlora-training/runs/RUN/checkpoint-exports/checkpoint-N.tar
```

The launcher validates the archive before allocating compute, uploads it
separately from the code/data bundle in retryable 32 MiB parts, reassembles and
SHA-256 verifies it remotely, extracts it with traversal checks, and requires the
trainer to restore the newest checkpoint. Google Drive is therefore not required
for checkpoint durability in the CLI workflow.

## Full ablation runs

After the smoke test passes:

```bash
# Natural-distribution Qwen3 control
bash scripts/run_colab_qlora_sft.sh \
  --model qwen3-4b-instruct-2507 --dataset base

# Feature-aware Qwen3 main run
bash scripts/run_colab_qlora_sft.sh \
  --model qwen3-4b-instruct-2507 --dataset curriculum
```

Train the smaller comparison models on the curriculum only after the Qwen3
ablation establishes whether the curriculum helps:

```bash
bash scripts/run_colab_qlora_sft.sh \
  --model qwen2.5-coder-1.5b-instruct --dataset curriculum

bash scripts/run_colab_qlora_sft.sh \
  --model deepseek-coder-1.3b-instruct --dataset curriculum
```

Each adapter must be evaluated with the unchanged 1,034-example generation and
SQLite execution harness. Validation loss is diagnostic and is not the primary
selection metric.

## Method references

- Hugging Face Transformers bitsandbytes documentation:
  https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Hugging Face PEFT LoRA reference:
  https://huggingface.co/docs/peft/en/package_reference/lora
- Hugging Face TRL SFT data and loss-mask documentation:
  https://huggingface.co/docs/trl/sft_trainer
