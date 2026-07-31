# Qwen3-4B decoding search

All sampled runs use the seed-17 selected QLoRA adapter, four candidates, top-p 0.95, and execution-consensus selection unless noted.

| Temperature / mode | Set | Strict EX | Compatible EX | Oracle | Mean generation |
|---|---:|---:|---:|---:|---:|
| Greedy | 300 | 86.000% | 89.000% | n/a | n/a |
| 0.2 | 300 | 86.000% | 89.000% | n/a | n/a |
| **0.5** | **300** | **87.667%** | **90.667%** | 89.667% | 2,645 ms/example |
| 0.8 | 300 | 85.667% | 88.667% | 90.333% | 2,858 ms/example |
| 0.2 | 1,001 | 86.613% | 89.211% | 87.812% | 2,424 ms/example |
| **0.5** | **1,001** | **87.512%** | **90.010%** | **90.110%** | **2,495 ms/example** |
| Greedy | 1,001 | 86.513% | 89.111% | n/a | 990 ms/example |

Temperature 0.5 is the confirmed accuracy mode. Against full-set greedy it gains 0.999 strict point and 0.899 compatible point, with approximately 2.5 times the observed latency. Greedy remains the default latency-oriented mode.

Value-aware voting was evaluated offline using the same candidates. It scored 86.667% strict at temperature 0.5 and 86.333% at temperature 0.8 on the 300-example screen, so neither beat temperature-0.5 execution consensus. Temperature 1.0 produced no scientific result because three Colab CLI websocket attempts failed before generation began.
