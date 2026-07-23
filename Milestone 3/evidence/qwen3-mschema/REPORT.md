# Qwen3 base SFT: DDL versus M-Schema prompt

| Variant | Execution | Normalized exact | Syntax valid |
|---|---:|---:|---:|
| qwen3-base-sft-ddl | 76.886% (795/1034) | 50.193% | 100.000% |
| qwen3-base-sft-mschema | 78.627% (813/1034) | 48.743% | 100.000% |

## Paired change

59 corrected, 41 regressed, net +18 (+1.741 pp); exact McNemar p=0.0886261.

## Complexity

| Slice | N | Before | After | Delta |
|---|---:|---:|---:|---:|
| simple | 593 | 84.486% | 85.666% | +1.180 pp |
| moderate | 336 | 71.131% | 72.619% | +1.488 pp |
| complex | 105 | 52.381% | 58.095% | +5.714 pp |

Execution is result equivalence on the supplied read-only SQLite databases.
