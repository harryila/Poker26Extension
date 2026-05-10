# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_bet_or_raise` (n=10)
- Target bucket: `clean_legal_fold` (n=30)
- Layers: [12, 13, 14, 15, 18]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': 1.2644980713403087, 'n': 5}, '13': {'mean_delta': 2.1158414994296812, 'n': 5}, '14': {'mean_delta': 5.234599375635768, 'n': 5}, '15': {'mean_delta': 7.4113745278875225, 'n': 5}, '18': {'mean_delta': 9.302136560367313, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 5.234599375635768
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 300 | +1.414 | +1.264 | **+0.149** | 0.3% | 99.7% | 0.0% |
| 13 | 300 | +2.781 | +2.116 | **+0.665** | 3.7% | 96.3% | 0.0% |
| 14 | 300 | +4.741 | +5.235 | **-0.494** | 4.0% | 24.7% | 71.3% |
| 15 | 300 | +6.675 | +7.411 | **-0.736** | 1.0% | 0.0% | 99.0% |
| 18 | 300 | +6.645 | +9.302 | **-2.657** | 9.0% | 0.0% | 91.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
