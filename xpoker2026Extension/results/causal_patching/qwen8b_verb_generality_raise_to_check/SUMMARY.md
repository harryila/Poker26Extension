# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_bet_or_raise` (n=10)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [18, 20, 22, 24, 30]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'18': {'mean_delta': -0.30000003051990093, 'n': 5}, '20': {'mean_delta': -2.299999889232306, 'n': 5}, '22': {'mean_delta': -2.599991951749582, 'n': 5}, '24': {'mean_delta': -2.174990077970873, 'n': 5}, '30': {'mean_delta': -4.622785063621382, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -2.599991951749582
- `random_source_test_layer` = 22

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 300 | -0.195 | -0.300 | **+0.105** | 100.0% | 0.0% | 0.0% |
| 20 | 300 | -2.337 | -2.300 | **-0.037** | 100.0% | 0.0% | 0.0% |
| 22 | 300 | -2.788 | -2.600 | **-0.188** | 93.0% | 0.0% | 7.0% |
| 24 | 300 | -5.708 | -2.175 | **-3.533** | 60.3% | 0.0% | 39.7% |
| 30 | 300 | -13.831 | -4.623 | **-9.208** | 0.7% | 0.0% | 99.3% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
