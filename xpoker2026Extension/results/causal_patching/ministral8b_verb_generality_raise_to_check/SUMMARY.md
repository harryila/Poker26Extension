# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched logs (pooled, n=3):
  - `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_bet_or_raise` (n=7)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [12, 14, 15, 16, 20]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': -0.2999747912353129, 'n': 5}, '14': {'mean_delta': -2.8498375600949153, 'n': 5}, '15': {'mean_delta': -6.924484991748696, 'n': 5}, '16': {'mean_delta': -9.498142496596415, 'n': 5}, '20': {'mean_delta': -11.016753989101996, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -6.924484991748696
- `random_source_test_layer` = 15

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 210 | -0.370 | -0.300 | **-0.070** | 100.0% | 0.0% | 0.0% |
| 14 | 210 | -0.212 | -2.850 | **+2.638** | 100.0% | 0.0% | 0.0% |
| 15 | 210 | +0.736 | -6.924 | **+7.661** | 95.7% | 0.0% | 4.3% |
| 16 | 210 | -0.318 | -9.498 | **+9.180** | 4.3% | 0.0% | 95.7% |
| 20 | 210 | -1.146 | -11.017 | **+9.871** | 0.0% | 0.0% | 100.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
