# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched log: `logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl.gz`
- **Residual-top-1 target filter**: kept only targets whose baseline residual top-1 was in family `FOLD` (6/30 = 20.0% retained).
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `clean_legal_fold` (n=6)
- Layers: [8, 10, 12, 14, 16, 18, 20]

## Controls
- `baseline_top1_match_rate` = 0.2
- `target_residual_top1_filter` = {'family': 'FOLD', 'n_before': 30, 'n_after': 6, 'frac_kept': 0.2}
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'8': {'mean_delta': -0.049935651527036384, 'n': 5}, '10': {'mean_delta': 0.0063511955544619525, 'n': 5}, '12': {'mean_delta': 0.2842303640012432, 'n': 5}, '14': {'mean_delta': 0.6351184240370287, 'n': 5}, '16': {'mean_delta': 0.9636931057163871, 'n': 5}, '18': {'mean_delta': 1.399892210985184, 'n': 5}, '20': {'mean_delta': 1.2228516908643328, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 0.6351184240370287
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 60 | -0.016 | -0.050 | **+0.034** | 0.0% | 100.0% | 0.0% |
| 10 | 60 | +0.024 | +0.006 | **+0.018** | 3.3% | 96.7% | 0.0% |
| 12 | 60 | +0.048 | +0.284 | **-0.237** | 11.7% | 88.3% | 0.0% |
| 14 | 60 | +0.259 | +0.635 | **-0.376** | 50.0% | 50.0% | 0.0% |
| 16 | 60 | +0.530 | +0.964 | **-0.433** | 76.7% | 23.3% | 0.0% |
| 18 | 60 | +0.786 | +1.400 | **-0.613** | 80.0% | 20.0% | 0.0% |
| 20 | 60 | +0.718 | +1.223 | **-0.505** | 76.7% | 23.3% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
