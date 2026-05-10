# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_legal_fold` (n=10)
- Target bucket: `clean_bet_or_raise` (n=30)
- Layers: [12, 13, 14, 15, 18]

## Controls
- `baseline_top1_match_rate` = 0.9666666666666667
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': 0.16772626945899205, 'n': 5}, '13': {'mean_delta': 1.7960234477566686, 'n': 5}, '14': {'mean_delta': 6.563144859555626, 'n': 5}, '15': {'mean_delta': 8.264145610307587, 'n': 5}, '18': {'mean_delta': 8.856006256909808, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 6.563144859555626
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 300 | +0.054 | +0.168 | **-0.113** | 2.3% | 0.0% | 97.7% |
| 13 | 300 | +0.095 | +1.796 | **-1.701** | 1.7% | 0.0% | 98.3% |
| 14 | 300 | -1.652 | +6.563 | **-8.216** | 4.7% | 8.0% | 87.3% |
| 15 | 300 | -5.697 | +8.264 | **-13.962** | 0.0% | 100.0% | 0.0% |
| 18 | 300 | -7.526 | +8.856 | **-16.382** | 0.0% | 100.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
