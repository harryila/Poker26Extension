# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched logs (pooled, n=3):
  - `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_legal_fold` (n=10)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [12, 16, 20, 26, 30]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': -0.3499847352456641, 'n': 5}, '16': {'mean_delta': -7.672679618866678, 'n': 5}, '20': {'mean_delta': -8.421047727685226, 'n': 5}, '26': {'mean_delta': -9.29162042576273, 'n': 5}, '30': {'mean_delta': -9.464922138157288, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -8.421047727685226
- `random_source_test_layer` = 20

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 300 | -0.233 | -0.350 | **+0.117** | 100.0% | 0.0% |
| 16 | 300 | -10.004 | -7.673 | **-2.331** | 0.0% | 100.0% |
| 20 | 300 | -12.021 | -8.421 | **-3.600** | 0.0% | 100.0% |
| 26 | 300 | -14.720 | -9.292 | **-5.429** | 0.0% | 100.0% |
| 30 | 300 | -15.395 | -9.465 | **-5.930** | 0.0% | 100.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation.
