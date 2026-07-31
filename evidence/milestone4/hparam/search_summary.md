# Qwen3 QLoRA hyperparameter search

Selection uses database-disjoint generated-SQL strict execution accuracy; loss is diagnostic.

| Trial | LR | r/alpha | Dropout | Train rows | Eval loss | Strict EX | Compatible EX | Syntax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cap-r16-a32-d0-all | 0.0003 | 16/32 | 0.0 | 2048 | pending | 85.814 | 88.611 | 100.0 |
| cap-r16-a32-d05-attn | 0.0003 | 16/32 | 0.05 | 2048 | pending | 85.514 | 88.312 | 100.0 |
| cap-r16-a32-d10-all | 0.0003 | 16/32 | 0.1 | 2048 | pending | 86.014 | 89.111 | 100.0 |
| cap-r32-a64-d05-all | 0.0003 | 32/64 | 0.05 | 2048 | pending | 85.514 | 88.312 | 100.0 |
| cap-r8-a16-d05-all | 0.0003 | 8/16 | 0.05 | 2048 | pending | 86.114 | 88.811 | 100.0 |
| lr1e4 | 0.0001 | 16/32 | 0.05 | 2048 | pending | 84.815 | 87.413 | 100.0 |
| lr2e4 | 0.0002 | 16/32 | 0.05 | 2048 | pending | 86.214 | 89.211 | 100.0 |
| lr3e4 | 0.0003 | 16/32 | 0.05 | 2048 | 0.215 | 86.513 | 89.111 | 100.0 |
| lr5e5 | 5e-05 | 16/32 | 0.05 | 2048 | 0.226835 | 83.117 | 86.414 | 100.0 |
| opt-cosine-w10-e1 | 0.0003 | 16/32 | 0.05 | 2048 | pending | 84.815 | 87.712 | 100.0 |
| opt-linear-w03-e1 | 0.0003 | 16/32 | 0.05 | 2048 | pending | 86.314 | 89.311 | 100.0 |
| stability-seed29 | 0.0003 | 16/32 | 0.05 | 2048 | pending | 86.214 | 89.61 | 100.0 |
| stability-seed41 | 0.0003 | 16/32 | 0.05 | 2048 | pending | 85.514 | 88.711 | 99.8 |
