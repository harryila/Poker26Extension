# Tier 1A.small CoT pilot — summary

Same seed (42), same temps (0.0 and 0.2), same opponent (informative_v2), 50 hands/cell.
Only difference: --cot flag.

| Model | Mode | Decisions | Parse rate | Trash one-hot (of parsed) | Mean reasoning chars |
|---|---|---:|---:|---:|---:|
| llama8b | no_cot | 3319 | 100.0% | 100.0% | 0 |
| llama8b | cot | 600 | 99.7% | 0.6% | 809 |
| llama8b | **delta** | | -0.3p | -99.4p | |
| qwen8b | no_cot | 979 | 100.0% | 100.0% | 0 |
| qwen8b | cot | 1017 | 1.3% | 0.0% | 328 |
| qwen8b | **delta** | | -98.7p | -100.0p | |
| ministral8b | no_cot | 844 | 5.0% | 0.0% | 0 |
| ministral8b | cot | 517 | 99.0% | 0.0% | 552 |
| ministral8b | **delta** | | +94.1p | +0.0p | |
