# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_bet_or_raise` (n=10)
- Target bucket: `clean_legal_fold` (n=30)
- Layers: [18, 20, 22, 24, 30]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'18': {'mean_delta': 0.8188011724418403, 'n': 5}, '20': {'mean_delta': 7.820605750312936, 'n': 5}, '22': {'mean_delta': 16.084523065192048, 'n': 5}, '24': {'mean_delta': 26.634523042391486, 'n': 5}, '30': {'mean_delta': 35.03452303875541, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 16.084523065192048
- `random_source_test_layer` = 22

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 300 | +1.555 | +0.819 | **+0.736** | 0.0% | 100.0% | 0.0% |
| 20 | 300 | +7.606 | +7.821 | **-0.215** | 2.3% | 97.7% | 0.0% |
| 22 | 300 | +22.011 | +16.085 | **+5.926** | 28.7% | 21.7% | 49.7% |
| 24 | 300 | +27.062 | +26.635 | **+0.427** | 21.3% | 0.0% | 78.7% |
| 30 | 300 | +29.803 | +35.035 | **-5.231** | 0.3% | 0.0% | 99.7% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
