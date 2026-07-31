# Qwen3-4B Text-to-SQL Hyperparameter Tuning Report

## Executive conclusion

The selected QLoRA training configuration is **learning rate 3e-4, rank 16, alpha 32, dropout 0.05, all linear projection layers, cosine scheduling, 3% warmup, one epoch, and seed 17 for the final reproducible run**. It achieved **86.513% strict execution accuracy**, **89.111% compatible execution accuracy**, and **100% SQL syntax validity** on a fixed 1,001-example database-disjoint tuning set.

The configuration was repeated with seeds 29 and 41. Across the three seeds, strict execution accuracy averaged **86.080%** with a **0.999 percentage-point range**; compatible execution averaged **89.144%**. This shows that the selected region is stable enough for a final full-data training run, although the seed-to-seed spread should still be reported.

For inference, greedy decoding remains the recommended default when latency and cost matter. Four-candidate sampling at temperature 0.5 improved strict execution on the common 300-example decoding screen from **86.000% to 87.667%**, while temperature 0.2 tied greedy on that screen and temperature 0.8 fell to 85.667%. On the full 1,001-example tuning set, temperature 0.5 achieved **87.512% strict execution** and **90.010% compatible execution**: improvements of 0.999 and 0.899 percentage point over greedy, respectively.

## Experimental question and leakage controls

The goal was not to find the lowest language-model loss. It was to select the QLoRA and decoding settings that maximize executable SQL correctness for `Qwen/Qwen3-4B-Instruct-2507` under an NVIDIA L4/Google Colab budget.

Hyperparameter selection used only databases derived from the Spider training split. The official Spider validation split was excluded from this HPO study and from gradient training. However, it had already been examined in Milestone 3 architecture, prompt, and ensemble experiments, so it is not a globally untouched test set and must not be presented as an unbiased final benchmark.

| Partition | Rows | Databases | Role |
|---|---:|---:|---|
| Internal training pool | 5,996 | 120 | Finalist training and future full-budget training |
| Screening training subset | 2,048 | 120 | Fast, consistent HPO budget |
| Internal tuning set | 1,001 | 20 | Model selection and stability evaluation |
| Decoding screen | 300 | 20 | Proportional subset of the internal tuning set |

There is zero database overlap between the 120 training databases and 20 tuning databases. The 2,048-row screening set preserves the original complexity distribution using proportional quotas and database round-robin sampling: 1,169 simple, 656 moderate, and 223 complex examples. The 300-row decoding screen similarly contains 172 simple, 96 moderate, and 32 complex examples.

Each example supplies the natural-language question, database identifier, M-Schema representation with representative values, optional evidence, and target SQL. The assistant loss is applied only to target SQL tokens.

## Fixed training setup

Unless the named factor was being varied, all trials used the following settings:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Pinned revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Logical parameters | 4,022,468,096 |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | BF16 |
| Attention | PyTorch SDPA |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear, no bias |
| Optimizer | paged AdamW 8-bit |
| Learning rate | 3e-4 |
| Scheduler / warmup | cosine / 0.03 |
| Epochs | 1 (128 optimizer steps) |
| Micro-batch / accumulation | 2 / 8 |
| Effective batch size | 16 |
| Maximum sequence length | 4,096 |
| Gradient checkpointing | enabled, non-reentrant |
| Gradient clipping | 0.3 |
| Weight decay | 0.0 |
| Default seed | 17 |
| Hardware | NVIDIA L4, 22.03 GiB VRAM |

The selected r=16 all-linear adapter has 33,030,144 trainable parameters, or 0.821% of the logical base model. Its compact adapter artifact is approximately 148 MB. The base weights are never stored on the Mac: every Colab VM downloads the pinned public checkpoint directly from Hugging Face into ephemeral `/content` storage. Only adapters, configuration, metrics, predictions, logs, and session notebooks are downloaded locally.

## Search method

The search was a controlled, factor-by-factor screening study rather than an unstructured grid. This was chosen because every 2,048-row L4 training run takes roughly 34–48 minutes, and the Colab account exposed only two concurrent L4 sessions despite a requested cap of six.

The primary selection metric was strict execution accuracy on all 1,001 internal tuning examples. Compatible execution accuracy, syntax validity, complexity-level accuracy, storage, VRAM, and latency were secondary metrics. Training loss was diagnostic only and was intentionally skipped in most later screening trials to save compute.

### Learning rate

| Learning rate | Strict EX | Compatible EX | Syntax |
|---:|---:|---:|---:|
| 5e-5 | 83.117% | 86.414% | 100.0% |
| 1e-4 | 84.815% | 87.413% | 100.0% |
| 2e-4 | 86.214% | 89.211% | 100.0% |
| **3e-4** | **86.513%** | 89.111% | 100.0% |

The primary metric peaked at 3e-4. The 2e-4 result was only 0.299 point lower and 0.100 point higher under compatible scoring, so both lie in a strong region; 3e-4 was retained because strict execution was declared primary before the search.

### LoRA capacity and target layers

| Configuration | Trainable parameters | Adapter size | Strict EX | Compatible EX |
|---|---:|---:|---:|---:|
| r=8, alpha=16, all-linear | 16.52M | 82.0 MB | 86.114% | 88.811% |
| **r=16, alpha=32, all-linear** | **33.03M** | **148.1 MB** | **86.513%** | **89.111%** |
| r=32, alpha=64, all-linear | 66.06M | 280.2 MB | 85.514% | 88.312% |
| r=16, alpha=32, attention-only | 11.80M | 63.1 MB | 85.514% | 88.312% |

Rank 16 was the best accuracy/capacity point. Rank 32 doubled trainable parameters and adapter storage but reduced strict accuracy by 0.999 point. Attention-only adaptation is attractive for storage-constrained deployment, but all-linear adaptation improved strict accuracy by the same 0.999 point.

### Regularization, scheduler, and warmup

| Factor | Setting | Strict EX | Compatible EX |
|---|---|---:|---:|
| Dropout | 0.00 | 85.814% | 88.611% |
| Dropout | **0.05** | **86.513%** | **89.111%** |
| Dropout | 0.10 | 86.014% | 89.111% |
| Scheduler | linear | 86.314% | **89.311%** |
| Scheduler | **cosine** | **86.513%** | 89.111% |
| Warmup | **0.03** | **86.513%** | **89.111%** |
| Warmup | 0.10 | 84.815% | 87.712% |

Dropout 0.05 gave the best strict score. Cosine narrowly beat linear by 0.199 point on the primary metric, while linear was 0.200 point better on compatible execution. A 10% warmup was clearly too conservative for a 128-step run and lost 1.698 strict points.

### Epoch budget

The two-epoch trial reached step 163/256 and then stopped making progress for more than 11 minutes. Because screening checkpoints were intentionally disabled to avoid large transient artifacts, no valid final adapter could be recovered. The run was terminated to prevent wasted L4 allocation and is recorded as an incomplete infrastructure trial, not as a scored configuration. No claim is made that two epochs are intrinsically worse; only that the attempted run did not produce comparable evidence. One epoch is therefore selected for the current reproducible configuration.

### Seed stability

| Seed | Strict EX | Compatible EX | Syntax |
|---:|---:|---:|---:|
| 17 | 86.513% | 89.111% | 100.0% |
| 29 | 86.214% | 89.610% | 100.0% |
| 41 | 85.514% | 88.711% | 99.8% |
| Mean | **86.080%** | **89.144%** | 99.93% |

The strict population standard deviation is 0.419 point and the range is 0.999 point; compatible execution has a 0.368-point population standard deviation. Seed 17 is kept as the reproducible final seed because it was the predeclared default and also produced the highest strict score; seeds are not treated as hyperparameters to cherry-pick.

## Decoding search

Temperature does not affect greedy decoding. It matters only when sampling multiple candidates. The decoding study generated four candidates per question with top-p 0.95, then used execution-consensus selection: execute all candidates, group candidates with identical executable result signatures, and choose the earliest member of the largest cluster. Selection does not inspect gold SQL or gold results.

| Strategy | Evaluation set | Strict EX | Compatible EX | Candidate oracle | Result |
|---|---:|---:|---:|---:|---|
| Greedy, one candidate | 300 screen | 86.000% | 89.000% | n/a | Fast reference |
| Temperature 0.2, four candidates | 300 screen | 86.000% | 89.000% | not used | No screen gain |
| **Temperature 0.5, four candidates** | **300 screen** | **87.667%** | **90.667%** | **89.667%** | Screen winner |
| Temperature 0.8, four candidates | 300 screen | 85.667% | 88.667% | 90.333% | Rejected by deployable score |
| Temperature 1.0, four candidates | 300 screen | not scored | not scored | not scored | Three CLI websocket failures before generation |
| Temperature 0.2, four candidates | 1,001 full tune | 86.613% | 89.211% | 87.812% | Only +0.100 over greedy |
| **Temperature 0.5, four candidates** | **1,001 full tune** | **87.512%** | **90.010%** | **90.110%** | **Confirmed accuracy mode** |

Offline selector comparisons reused the exact stored candidates, avoiding regeneration. At temperature 0.5, FINER-style value-aware voting scored 86.667% strict and 90.333% compatible, versus 87.667% and 90.667% for execution consensus. At temperature 0.8 it improved strict execution from 85.667% to 86.333%, but still remained below temperature-0.5 execution consensus. Execution consensus at temperature 0.5 is therefore retained.

Temperature 0.8 produced the highest screen oracle but the worst deployable selected score. This means it created a more diverse pool containing good SQL more often, but execution consensus could not reliably identify that SQL. It may be useful in a future learned-verifier experiment, but the oracle number cannot justify deployment by itself.

Four-candidate inference takes approximately 2.4–2.9 seconds per example in the tested batch configuration and reached about 13.8 GiB peak allocated VRAM. The full temperature-0.5 confirmation measured 2.495 seconds per example. Greedy inference took about 0.99 second per example and peaked at 10.7 GiB allocated VRAM. Sampling therefore performs four candidate generations but costs roughly 2.5 times the observed wall-clock latency because candidates are batched. Consequently:

- Use **greedy decoding** as the default low-latency mode.
- Use **temperature 0.5, top-p 0.95, four candidates, execution consensus** as the confirmed accuracy-oriented mode when the 0.999-point strict gain justifies roughly 2.5 times the latency.
- Treat candidate oracle accuracy only as an upper bound; it uses knowledge of whether any candidate matches the gold execution and is not deployable.

## Final selected configuration

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
  "method": "QLoRA",
  "quantization": "NF4 4-bit, double quantization, BF16 compute",
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": "all-linear",
  "learning_rate": 0.0003,
  "optimizer": "paged_adamw_8bit",
  "scheduler": "cosine",
  "warmup_ratio": 0.03,
  "epochs": 1,
  "micro_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 16,
  "max_sequence_length": 4096,
  "seed": 17
}
```

This configuration was selected using only the database-disjoint internal tuning set. It was then retrained once on the full 5,996-row internal training pool, confirmed on the same 1,001-example internal set without further tuning, and evaluated once with locked greedy decoding on the official Spider validation split. That split was never folded into gradient training or used to change the selected HPO configuration, but it had been used for earlier Milestone 3 system development and is therefore not globally untouched.

## Full-data retraining and locked final evaluation

The selected configuration was retrained on all 5,996 approved internal examples across 120 training databases. Colab runtime preemptions required verified checkpoint resumes at steps 100, 200, 300, and 350; the final step-375 adapter matches the final checkpoint byte-for-byte. Resume state included optimizer, scheduler, RNG, and trainer state, so these were continuations of one run rather than restarts. Artifact validation confirmed 33,030,144 trainable adapter parameters, zero token truncation, final evaluation loss 0.222494, and adapter SHA-256 `63a51ff491a163c1433dd4ac56d936d969337de67279f852ea1ef966ac335e5c`.

| Evaluation | Examples | Strict EX | Compatible EX | Syntax | Normalized EM |
|---|---:|---:|---:|---:|---:|
| Internal database-disjoint confirmation | 1,001 | 85.514% | 89.011% | 100.000% | 51.349% |
| Official Spider validation (previously used in M3 development) | 1,034 | 77.853% | 81.238% | 99.903% | 50.387% |

The full-data internal confirmation was 0.999 strict point below the 2,048-row screening adapter. This is not evidence that hyperparameter tuning failed: within the controlled 2,048-row screen, tuning improved strict execution by 3.396 points from 83.117% at 5e-5 to 86.513% at the selected configuration. The full-data adapter still scored 2.397 points above that weakest screening result, but that cross-budget difference is not attributed solely to hyperparameters because the training-row count also changed. More training rows did not automatically preserve the screening peak, so the full-data result is reported transparently rather than replaced with the better-looking screening number.

Official compatible execution by complexity was 88.702% for simple, 72.917% for moderate, and 65.714% for complex queries. These databases and schemas were unseen by the final adapter during gradient training, but their split-level results had been observed during earlier project development. The gap reinforces that internal tuning scores are selection estimates and that a new frozen holdout is required for final test claims. No decoding or training choice was changed after the final adapter's official-split score was observed.

## Compute, artifacts, and reproducibility

Peak allocated training VRAM was about 16.9 GiB for the selected adapter; peak reserved memory was about 20.4 GiB on the 22.03-GiB L4. A representative selected run trained for about 2,885 seconds and performed its optional loss evaluation in about 271 seconds. Later trials skipped full loss evaluation because execution accuracy, not loss, determined selection.

Primary artifacts:

- `configs/hparam/qwen3/`: exact per-trial training configurations.
- `artifacts/qlora-hparam/search_summary.{md,csv,json}`: machine- and human-readable trial aggregation.
- `artifacts/qlora-hparam/runs/`: compact adapters, manifests, metrics, and orchestration logs.
- `artifacts/qlora-hparam/macsql/`: compatible execution rescoring.
- `artifacts/qlora-hparam/decoding/`: decoding screens and alternate-selector results.
- `data/finetuning/qwen3_hparam_mschema_v1/`: database-disjoint train/tune package and decoding screen.
- `scripts/run_colab_qlora_hparam.sh`: training orchestration with automatic session shutdown.
- `scripts/run_colab_zero_shot_eval.sh`: evaluation and sampled decoding orchestration with automatic session shutdown.
- `scripts/summarize_qwen3_hparam_search.py`: result aggregation.
- `scripts/build_qwen3_decode_screen.py`, `scripts/filter_predictions_by_validation.py`, and `scripts/reselect_text2sql_candidates.py`: decoding-study utilities.

## Limitations

- This is a controlled one-factor-at-a-time search, not an exhaustive factorial or Bayesian optimization study.
- Most training configurations were screened on 2,048 rows, so rankings could shift after full-data training.
- Differences below roughly one percentage point should be interpreted cautiously given the observed seed spread.
- The two-epoch result is missing because the run did not complete; it is not evidence against longer training.
- The temperature-1.0 run has no scientific result because generation never started.
- Execution accuracy can accept semantically equivalent SQL, while normalized exact match is stricter about surface form; both are retained in raw metrics.
- Multi-candidate execution assumes safe, read-only, resource-limited database execution in deployment.

## Decision

Freeze the step-375 full-data adapter as the reproducible production artifact. Keep greedy decoding as the deployment default. The four-candidate temperature-0.5 execution-consensus mode remains an internally confirmed optional accuracy mode, but it was deliberately not rerun on official Spider validation after the final greedy evaluation. Future work should use the stored predictions for error analysis and must not retune against official validation.
