## Milestone 4 Report: Model Training, Optimization, and Selection

Talk to Your Database — A Natural-Language Analytics Copilot

- Course: Data Science and AI Lab

- Milestone: 4 — Model Training and Optimization

- Submission date: July 30, 2026

Team members: Siddhant Hitesh Mantri (21f3002218), Anirudh Komanduri (22f1000522),

Vishal S (23f2003089), Walunila Aier (21f3002564), Sambhav Jha (22f3003227), and Smrutishikta Das (21f1006009)

## Executive Summary

Milestone 4 optimized the Milestone 3 choice, Qwen/Qwen3-4B-Instruct-2507 , using QLoRA on one Colab L4. Thirteen configurations were compared on a database-disjoint Spider split. The winning 2,048-row screening run used learning rate 3e-4 , rank/alpha 16/32, dropout 0.05, all-linear targets, cosine scheduling, 3% warmup, effective batch 16, one epoch, and seed 17; it achieved 86.513% strict execution, 89.111% compatible execution, and 100% syntax validity.

The recipe was then retrained on all 5,996 internal training examples. Final checkpoint 375 achieved 85.514% strict, 89.011% compatible, and 100% syntax validity on 1,001 tuning examples. Its adapter trains 33.03M parameters (0.821% of the 4.022B base). Spider validation was excluded from gradient training and HPO but had been examined in Milestone 3, so Milestone 5 requires a newly frozen holdout and/or BIRD for unbiased claims.

## 1. Introduction

The project converts a natural-language question and database schema into one safe, read-


only SQL query. The current pipeline supplies the model with schema information, generates SQL, parses and validates it, and executes it on an immutable SQLite connection.

Qwen3-4B was selected in Milestone 3 because it was the strongest original general-model baseline, remained within the 4B limit, supported later instruction-following tasks, and could be adapted on one NVIDIA L4.

The objectives of Milestone 4 were to:

- 1. create a leakage-controlled train/tuning protocol;

- 2. optimize the QLoRA training recipe;

- 3. study training stability and convergence;

- 4. retrain the selected recipe on the complete approved training pool;

- 5. freeze a reproducible final adapter for Milestone 5 evaluation.

## 2. Training Dataset

## 2.1 Source and preprocessing

The source is Spider 1.0 paired with the project's local SQLite databases. Three non- executable annotations were removed from the original 7,000 training rows, leaving 6,997 usable examples.

Preprocessing validated each target against its database, retained only read-only SQL, extracted schema structure and bounded values, rendered M-Schema, created chat turns, masked prompt tokens from loss, and recorded checksums and metadata. Overlength examples were rejected rather than silently truncated.

## 2.2 Final split

The 6,997 usable Spider training examples were divided by database, not by random rows:

| Partition |   |   | Examples Databases Simple Moderate Complex |   |   | Use |
| --- | --- | --- | --- | --- | --- | --- |
| Internal | 5,996 | 120 | 3,423 | 1,921 | 652 | Gradient |


| Partition |   |   | Examples Databases Simple Moderate Complex |   |   | Use |
| --- | --- | --- | --- | --- | --- | --- |
| training |   |   |   |   |   | training |
| Internal tuning | 1,001 | 20 | 572 | 321 | 108 | HPO and selection |
| Official Spider validation | 1,034 | 20 | 593 | 336 | 105 | No gradient training; previously used in M3 |

There is zero database/schema overlap between internal training and tuning. For faster HPO, a fixed 2,048-row screen retained all 120 training databases and approximately preserved complexity proportions.

## 2.3 Input representation

Each example follows:

System: Generate one read-only SQLite query using only the supplied schema. User: Database dialect + M-Schema + natural-language question. Assistant: Gold SQL only.

For the final run, training sequences ranged from 255 to 3,173 tokens and tuning sequences from 291 to 1,476 tokens. The limit was 4,096 tokens; zero examples were truncated.

## 2.4 Data augmentation experiments

Two alternatives were tested before final HPO:

| Training data | Rows | Strict EX | Decision |
| --- | --- | --- | --- |
| Natural Spider | 6,997 | 76.886% | Retained |
| Feature-weighted curriculum | 9,796 | 74.371% Rejected |   |


| Training data | Rows | Strict EX | Decision |
| --- | --- | --- | --- |
| Natural + 5,000 filtered Gretel examples | 11,997 | 74.468% Rejected |   |

The curriculum overemphasized difficult templates and increased schema-linking errors.

Synthetic augmentation caused negative transfer, particularly on moderate queries. Therefore the final model used natural Spider data only.

## 3. Model Configuration

The final model is the pretrained decoder-only Qwen/Qwen3-4B-Instruct-2507 , pinned to revision:

## cdbee75f17c01a7cc42f958dc650907174af0554

| Property | Value |
| --- | --- |
| Logical parameters | 4,022,468,096 |
| Transformer layers | 36 |
| Hidden size | 2,560 |
| Intermediate size | 9,728 |
| Query / key-value heads | 32 / 8 |
| Project sequence limit | 4,096 |

The model was not trained from scratch. QLoRA retains the pretrained backbone, stores it in 4-bit NF4, and trains small low-rank adapters.

flowchart LR I[Question + M-Schema] --> T[Tokenizer and chat template] T --> B[Frozen Qwen3 backbone<br/>4-bit NF4 storage] A[Trainable rank-16<br/>LoRA matrices] --> B B --> H[Language-model head] H --> S[SQL tokens] S --> L[Assistant-only<br/>cross-entropy loss] L --> A


| QLoRA property | Final value |
| --- | --- |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | BF16 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | All linear layers |
| Trainable parameters | 33,030,144 |
| Trainable proportion | 0.821141% |
| Adapter weight size | 132,187,888 bytes |

The frozen backbone participates in forward and backward computation, but only LoRA parameters are updated.

## 4. Training Environment

Training used Google Colab sessions created and controlled through the official Colab CLI.

| Component | Final environment |
| --- | --- |
| GPU | NVIDIA L4 |
| Available VRAM | 22.03 GiB |
| Peak allocated / reserved | 16.98 / 20.48 GiB |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |
| PEFT | 0.19.1 |
| bitsandbytes | 0.49.2 |


| Component | Final environment |
| --- | --- |
| Accelerate | 1.14.0 |
| Attention | SDPA |

Base weights were downloaded into ephemeral Colab storage; only adapters, checkpoints, configurations, logs, predictions, and metrics were retained. Limited VRAM, runtime resets, websocket instability, and transfer failures motivated QLoRA, fixed-budget HPO, retryable transfers, and verified checkpoint recovery.

## 5. Training Methodology

flowchart TD D[Checksummed train/tuning JSONL] --> P[Tokenization preflight] P --> C[Create Colab L4] C --> M[Download pinned Qwen3] M --> Q[Load NF4 backbone + LoRA] Q --> T[Assistant-only supervised training] T --> K[Export and verify checkpoint] K -- >|Runtime reset| C K -->|Final step| V[Validation loss] V --> E[Generate SQL on tuning set] E -- > X[Strict and compatible execution scoring] X --> Z[Download artifacts and stop session]

## 5.1 Core settings

| Setting | Value |
| --- | --- |
| Per-device training batch | 2 |
| Gradient accumulation | 8 |
| Effective batch | 16 |
| Epochs | 1 |
| Final optimizer steps | 375 |
| Optimizer | Paged AdamW 8-bit |
| Maximum gradient norm | 0.3 |
| Gradient checkpointing | Enabled |


| Setting | Value |
| --- | --- |
| Seed | 17 |

Effective batch 16 was fixed across trials to fit the longest sequences and avoid confounding HPO factors; a batch-size sweep was not conducted.

## 5.2 Loss function

The model minimizes assistant-only causal cross-entropy:

Prompt tokens use label -100 . Execution accuracy was the selection metric because semantically equivalent SQL may differ textually.

## 5.3 Checkpointing and reproducibility

Verified checkpoints at steps 100, 200, 300, 350, and 375 contain the adapter, optimizer, scheduler, RNG and trainer states, tokenizer, and arguments, allowing exact resume. Runs also record the model revision, data checksums, package versions, seed, predictions, and GPU usage.

## 6. Hyperparameter Experiments

Thirteen complete configurations were trained on the identical 2,048-row screen and evaluated on all 1,001 tuning examples. Strict execution accuracy was the primary metric; compatible execution and syntax validity were secondary.


## 6.1 Factor sweep

Each group below varied one factor from the selected configuration:

| Factor | Variant | Strict EX | Compatible EX |
| --- | --- | --- | --- |
| Learning rate | 5e-5 | 83.117% | 86.414% |
|   | 1e-4 | 84.815% | 87.413% |
|   | 2e-4 | 86.214% | 89.211% |
|   | 3e-4 | 86.513% | 89.111% |
| LoRA | r8/a16, all-linear (16.52M) | 86.114% | 88.811% |
|   | r16/a32, all-linear (33.03M) | 86.513% | 89.111% |
|   | r32/a64, all-linear (66.06M) | 85.514% | 88.312% |


| Factor | Variant | Strict EX | Compatible EX |
| --- | --- | --- | --- |
|   | r16/a32, attention-only (11.80M) | 85.514% | 88.312% |
| Dropout | 0.00 | 85.814% | 88.611% |
|   | 0.05 | 86.513% | 89.111% |
|   | 0.10 | 86.014% | 89.111% |
| Scheduler | Linear, 3% warmup | 86.314% | 89.311% |
|   | Cosine, 3% warmup | 86.513% | 89.111% |
| Warmup | Cosine, 10% | 84.815% | 87.712% |

The selected learning rate improved strict execution by 3.396 points over 5e-5 . Rank 16 gave the best capacity/accuracy balance; rank 32 doubled adapter size but regressed. All- linear targets beat attention-only by 0.999 point. Dropout 0.05 was best, cosine and linear schedules were effectively tied, and 10% warmup was too long for a 128-step screen.

## 6.2 Epoch and seed checks

The two-epoch trial reached step 163/256 and then stopped progressing. No final adapter was recoverable, so no accuracy conclusion is claimed. One epoch was selected as the only fully validated epoch budget.

| Seed | Strict EX | Compatible EX | Syntax |
| --- | --- | --- | --- |
| 17 | 86.513% | 89.111% | 100.0% |
| 29 | 86.214% | 89.610% | 100.0% |
| 41 | 85.514% | 88.711% | 99.8% |
| Mean | 86.080% | 89.144% | 99.93% |

Strict execution had a 0.419-point population standard deviation and a 0.999-point range. Seed 17 was retained because it was predeclared, not selected after searching seeds.


## 6.3 Final decision

The selected recipe was:

LR 3e-4 + r16/alpha32 + dropout 0.05 + all-linear + cosine + 3% warmup + one epoch + effect

This produced the highest strict score, remained stable across seeds, avoided unnecessary rank-32 storage, and fit one L4.

## 7. Optimization Methods

Paged AdamW 8-bit provides adaptive transformer updates with memory-efficient optimizer states. The learning rate warms for about 11 of 375 steps and then follows cosine decay from 3e-4 ; eight micro-batches form each effective batch of 16. Non-reentrant gradient checkpointing reduces activation memory, gradient clipping at 0.3 limits unstable updates, BF16 accelerates arithmetic, and NF4 compresses frozen-weight storage.

Early stopping was not used because the selected run lasted one epoch and validation loss was evaluated once at the end. Future multi-epoch work should use a new internal development set rather than the previously reused Spider validation split.

## 8. Regularization Techniques

| Technique | Setting and justification |
| --- | --- |
| LoRA dropout | 0.05; best strict score among 0.00, 0.05, and 0.10 |
| Frozen backbone | Limits trainable capacity and catastrophic forgetting |
| Gradient clipping | 0.3; stabilizes update magnitude |
| Weight decay | 0.0; no evidence justified adding it |
| Label smoothing | 0.0; exact SQL token targets |


| Technique | Setting and justification |
| --- | --- |
| Data augmentation | Rejected after curriculum and synthetic regressions |
| Cross-validation | Not used because database-fold training would multiply GPU cost |
| Seed replication | Three seeds used as stability evidence |

Class weighting was not applicable to autoregressive generation. Holding out complete databases prevented schema leakage that a random row split could introduce.

## 9. Training Progress

The complete loss history was recovered from checkpoint 375, including steps completed before Colab resumes.


*Dashed vertical lines mark locally verified resumable checkpoints. Validation loss was evaluated only at the final epoch.*

Training completed 375/375 steps. Loss fell from 1.730 at step 1 to 0.325 by step 20 and then generally stayed around 0.08–0.24; learning rate and gradients followed the intended schedule. Final tuning loss was 0.222494. Because validation loss was measured only at epoch end, the single point suggests a generalization gap but cannot identify an ideal stopping step.

The final model was then evaluated on the 1,001-example tuning set:

| Model |   | Training rows Strict EX Compatible EX Syntax |   |   |
| --- | --- | --- | --- | --- |
| Selected screening adapter | 2,048 | 86.513% | 89.111% | 100% |
| Final checkpoint 375 | 5,996 | 85.514% | 89.011% | 100% |

The full-data run lost 0.999 strict point, confirming that more rows did not automatically preserve the screening peak and that execution—not loss alone—must drive evaluation.


## 10. Model Selection

Selection proceeded from the Milestone 3 architecture choice, to recipe selection by internal strict execution, to one full-data retraining. The release artifact is checkpoint 375:

artifacts/qlora-training/runs/20260727-083554-qwen3-4b-instruct-2507-18108/downloaded/outpu

It follows the predeclared procedure, uses all approved training examples, completed without divergence or truncation, is resumable, and matches checkpoint 375 byte-for-byte. This avoids choosing a checkpoint after observing tuning results.

Its adapter SHA-256 is:

## 63a51ff491a163c1433dd4ac56d936d969337de67279f852ea1ef966ac335e5c

The screening adapter remains the best internal-score adapter, while checkpoint 375 is the predeclared full-data release. Official Spider validation, candidate-oracle accuracy, token loss alone, and incomplete two-epoch evidence were excluded from selection.

## 11. Challenges Encountered

| Challenge | Mitigation |
| --- | --- |
| Full fine-tuning exceeded L4 feasibility | Used QLoRA, NF4, BF16, gradient checkpointing, and accumulation |
| Colab runtime resets | Exported and locally verified resumable checkpoints |
| 150–206 MB transfer failures | Used retryable 32 MiB multipart transfers and SHA-256 checks |
| Websocket interruptions | Added bounded retries, recovery downloads, and fresh-session relaunch |
| Limited parallel L4 capacity | Used fixed-budget factor-by-factor HPO |


| Challenge | Mitigation |
| --- | --- |
| Hyperparameter sensitivity | Compared LR, rank, target layers, dropout, scheduler, warmup, and seeds |
| Curriculum and synthetic negative transfer | Returned to natural-distribution Spider training |
| Official validation reuse in M3 | Created a separate internal HPO split and reserved a new holdout for M5 |

Uncompleted work—two epochs, batch-size and weight-decay sweeps, periodic validation, and cross-validation—is treated as limitation rather than negative evidence.

## 12. Summary and Next Steps

The selected Qwen3-4B QLoRA recipe uses 3e-4 , rank/alpha 16/32, dropout 0.05, all-linear targets, paged AdamW 8-bit, cosine decay, 3% warmup, effective batch 16, one epoch, and seed 17. Rank 16, all-linear adaptation, and moderate dropout performed best; long warmup and both tested augmentations regressed. Seed variation stayed within 0.999 strict point, while full-data retraining scored slightly below screening. The frozen revision, adapter, prompt, evaluator, predictions, and checksums make the system ready for comprehensive evaluation.

## Milestone 5 should:

- 1. evaluate the frozen adapters on an untouched database-disjoint holdout and/or BIRD;

- 2. report execution, syntax, exact match, latency, VRAM, and complexity slices;

- 3. test invalid, ambiguous, and out-of-domain requests;

- 4. avoid further tuning on the reused Spider validation split.


## Appendix A. Additional Work Outside the Core M4 Scope

Work outside core M4—including other models, prompt formats, consensus, FINER sampling, GradeSQL, safety infrastructure, and the post-freeze Spider run—is retained in the experiment ledger. Inference sampling is excluded from training HPO: four candidates at temperature 0.5/ top-p 0.95 with execution consensus reached 87.512% strict and 90.010% compatible, but cost about 2.5× greedy latency.

## References

- 1. Yu, T. et al. “Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross- Domain Semantic Parsing and Text-to-SQL Task.” EMNLP 2018. https://aclanthology.org/ D18-1425/ [URL 🔗](https://aclanthology.org/D18-1425/)

- 2. Qwen3-4B-Instruct-2507 model card. https://huggingface.co/Qwen/Qwen3-4B- Instruct-2507 [URL 🔗](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

- 3. Dettmers, T. et al. “QLoRA: Efficient Finetuning of Quantized LLMs.” https://arxiv.org/abs/ 2305.14314 [URL 🔗](https://arxiv.org/abs/2305.14314)

- 4. Hugging Face PEFT LoRA documentation. https://huggingface.co/docs/peft/en/ package_reference/lora [URL 🔗](https://huggingface.co/docs/peft/en/package_reference/lora)

- 5. XGenerationLab M-Schema. https://github.com/XGenerationLab/M-Schema [URL 🔗](https://github.com/XGenerationLab/M-Schema)

- 6. Google Developers Blog. “Introducing the Google Colab CLI.” June 5, 2026. https://developers.googleblog.com/introducing-the-google-colab-cli/ [URL 🔗](https://developers.googleblog.com/introducing-the-google-colab-cli/)

## SIGNATURE OF APPROVAL

- Siddhant Hitesh Mantri

- Anirudh Komanduri

- Vishal S

- Walunila Aier

- Sambhav Jha
