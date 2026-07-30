# Third-party data and models

This file is an attribution and review checklist, not legal advice.

| Component | Upstream location | Role |
|---|---|---|
| Spider 1.0 | `xlangai/spider` and the official Spider project | Training and validation benchmark |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | Selected base model |
| Qwen2.5-Coder-1.5B-Instruct | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Small-model baseline |
| DeepSeek-Coder-1.3B-Instruct | `deepseek-ai/deepseek-coder-1.3b-instruct` | Older/weak baseline |
| XiYanSQL-QwenCoder-3B-2504 | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` | SQL-specialist comparison |
| FINER-SQL-3B-Spider | `griffith-bigdata/FINER-SQL-3B-Spider` | Sampling/specialist comparison |
| GradeSQL Spider data | GradeSQL upstream release | Optional ORM verifier experiment |
| Gretel synthetic text-to-SQL | Gretel upstream release | Rejected augmentation experiment |

Spider-derived artifacts are documented as CC BY-SA 4.0 in the project data
pipeline. Before public redistribution, review the current license files and
model cards at each pinned upstream revision. This package does not redistribute
base-model weights, Spider databases, or full third-party datasets.

The team has not selected a license for its original source code in this
snapshot. Add an explicit project license before inviting general reuse.

