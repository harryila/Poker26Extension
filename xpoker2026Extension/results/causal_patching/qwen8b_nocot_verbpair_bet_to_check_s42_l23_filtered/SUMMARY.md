# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched log: `logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz`
- **Residual-top-1 target filter**: kept only targets whose baseline residual top-1 was in family `CHECK_CALL` (30/30 = 100.0% retained).
- **Residual-top-1 source filter**: kept only sources whose verb-position residual top-1 was in family `BET_RAISE` (10/10 = 100.0% retained).
- Source bucket: `clean_bet_or_raise` (n=10)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [23]

## Controls
- `source_residual_top1_filter` = {'family': 'BET_RAISE', 'n_before': 10, 'n_after': 10, 'frac_kept': 1.0}
- `baseline_top1_match_rate` = 1.0
- `target_residual_top1_filter` = {'family': 'CHECK_CALL', 'n_before': 30, 'n_after': 30, 'frac_kept': 1.0}
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'23': {'mean_delta': -9.774991832035713, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -9.774991832035713
- `random_source_test_layer` = 23

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 23 | 300 | -6.030 | -9.775 | **+3.745** | 0.0% | 0.0% | 100.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
