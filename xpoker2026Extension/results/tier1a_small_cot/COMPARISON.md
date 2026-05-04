# Tier 1A.small CoT vs no-CoT -- 18-cell cross-condition diff

Same 3 seeds (42, 123, 456), same 2 temps (0.0, 0.2), same opponent (informative_v2), same 100 hands/cell.
Only difference: --cot flag.

| Model | Mode | Cells | Decisions | Parse rate | Trash one-hot (of parsed) | Mean reasoning chars |
|---|---|---:|---:|---:|---:|---:|
| llama8b | no_cot | 6 | 8989 | 100.0% | 100.0% | 0 |
| llama8b | cot | 6 | 3446 | 99.7% | 1.5% | 815 |
| llama8b | **delta** | | | -0.3p | -98.5p | |
| qwen8b | no_cot | 6 | 3571 | 100.0% | 100.0% | 0 |
| qwen8b | cot | 6 | 3999 | 99.7% | 0.3% | 618 |
| qwen8b | **delta** | | | -0.3p | -99.7p | |
| ministral8b | no_cot | 6 | 1644 | 5.8% | 0.0% | 0 |
| ministral8b | cot | 6 | 1883 | 99.6% | 0.1% | 521 |
| ministral8b | **delta** | | | +93.8p | +0.1p | |

## Companion artifacts

- `pce_<tag>_summary.csv` -- clustered-bootstrap CIs on JS distances
  to CardOnly and StrategyAware oracles per model.
- `uc_<tag>_summary.json` -- update coherence metrics per model.
- `analyze_cot_<tag>.json` -- reasoning quality scores +
  JS-to-quality correlation per model.
- `pce_pool_summary.csv` -- pooled CoT PCE across all 3 models.
