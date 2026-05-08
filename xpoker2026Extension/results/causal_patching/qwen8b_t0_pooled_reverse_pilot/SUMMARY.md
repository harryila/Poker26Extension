# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_legal_fold` (n=10)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [16, 20, 24, 30, 34]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'16': {'mean_delta': 0.5250000020983421, 'n': 5}, '20': {'mean_delta': -1.0249999952291724, 'n': 5}, '24': {'mean_delta': -4.574999962724812, 'n': 5}, '30': {'mean_delta': -8.624075947026416, 'n': 5}, '34': {'mean_delta': -9.274614269246133, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -4.574999962724812
- `random_source_test_layer` = 24

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 300 | -0.226 | +0.525 | **-0.751** | 100.0% | 0.0% |
| 20 | 300 | -9.613 | -1.025 | **-8.588** | 100.0% | 0.0% |
| 24 | 300 | -30.763 | -4.575 | **-26.188** | 0.0% | 100.0% |
| 30 | 300 | -45.873 | -8.624 | **-37.249** | 0.0% | 100.0% |
| 34 | 300 | -46.860 | -9.275 | **-37.585** | 0.0% | 100.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation.
