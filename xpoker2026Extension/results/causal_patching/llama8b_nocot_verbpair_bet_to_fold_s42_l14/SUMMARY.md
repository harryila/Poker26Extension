# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched log: `logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl.gz`
- **Residual-top-1 target filter**: kept only targets whose baseline residual top-1 was in family `FOLD` (5/30 = 16.7% retained).
- Source bucket: `clean_bet_or_raise` (n=10)
- Target bucket: `clean_legal_fold` (n=5)
- Layers: [14]

## Controls
- `baseline_top1_match_rate` = 0.16666666666666666
- `target_residual_top1_filter` = {'family': 'FOLD', 'n_before': 30, 'n_after': 5, 'frac_kept': 0.16666666666666666}
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'14': {'mean_delta': 0.18215085084563754, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 0.18215085084563754
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 50 | +0.779 | +0.182 | **+0.597** | 90.0% | 10.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
