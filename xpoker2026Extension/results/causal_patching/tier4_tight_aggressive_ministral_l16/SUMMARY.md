# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched log: `logs/opp_tight_aggressive_ministral-8b_t00_s42_enriched.jsonl`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `clean_legal_fold` (n=30)
- Layers: [16]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'16': {'mean_delta': 0.7648335885603096, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 0.7648335885603096
- `random_source_test_layer` = 16

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 300 | +0.921 | +0.765 | **+0.156** | 12.7% | 49.0% | 38.3% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
