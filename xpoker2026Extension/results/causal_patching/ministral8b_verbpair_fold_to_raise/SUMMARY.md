# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched logs (pooled, n=3):
  - `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_legal_fold` (n=10)
- Target bucket: `clean_bet_or_raise` (n=7)
- Layers: [12, 14, 15, 16, 20]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': 0.12525426095876, 'n': 5}, '14': {'mean_delta': -0.24834184865926687, 'n': 5}, '15': {'mean_delta': -5.59832809080521, 'n': 5}, '16': {'mean_delta': -8.37207661962239, 'n': 5}, '20': {'mean_delta': -8.895830531387249, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -5.59832809080521
- `random_source_test_layer` = 15

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 70 | +0.195 | +0.125 | **+0.069** | 0.0% | 0.0% | 100.0% |
| 14 | 70 | -0.823 | -0.248 | **-0.574** | 0.0% | 0.0% | 100.0% |
| 15 | 70 | -5.868 | -5.598 | **-0.269** | 61.4% | 28.6% | 10.0% |
| 16 | 70 | -9.439 | -8.372 | **-1.067** | 0.0% | 100.0% | 0.0% |
| 20 | 70 | -11.243 | -8.896 | **-2.347** | 0.0% | 100.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
