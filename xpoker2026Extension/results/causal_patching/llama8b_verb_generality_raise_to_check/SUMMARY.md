# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_bet_or_raise` (n=10)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [12, 13, 14, 15, 18]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': 0.025352647170019792, 'n': 5}, '13': {'mean_delta': 0.12456402103304001, 'n': 5}, '14': {'mean_delta': -1.034742971623888, 'n': 5}, '15': {'mean_delta': -1.9959475623533778, 'n': 5}, '18': {'mean_delta': -2.2055338707042575, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -1.034742971623888
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 300 | -0.651 | +0.025 | **-0.676** | 100.0% | 0.0% | 0.0% |
| 13 | 300 | -1.361 | +0.125 | **-1.486** | 91.0% | 0.0% | 9.0% |
| 14 | 300 | -4.047 | -1.035 | **-3.013** | 55.7% | 0.0% | 44.3% |
| 15 | 300 | -6.298 | -1.996 | **-4.302** | 5.3% | 0.0% | 94.7% |
| 18 | 300 | -7.706 | -2.206 | **-5.501** | 8.3% | 0.0% | 91.7% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
