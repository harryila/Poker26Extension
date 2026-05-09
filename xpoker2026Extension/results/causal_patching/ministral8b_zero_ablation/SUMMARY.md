# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched logs (pooled, n=3):
  - `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- **Zero-ablation mode**: source residual is the all-zeros tensor (no real source sampled). The `--source-bucket` value below is used only to exclude that bucket from the random-null pool in control 3.
- Source bucket: `clean_check_or_call` (n=1)
- Target bucket: `illegal_fold` (n=30)
- Layers: [12, 14, 15, 16, 20]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': 0.14831208884200037, 'n': 5}, '14': {'mean_delta': 0.12896132635628277, 'n': 5}, '15': {'mean_delta': -0.3560074414099219, 'n': 5}, '16': {'mean_delta': -0.4025315616636881, 'n': 5}, '20': {'mean_delta': -1.0624754938317686, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -0.3560074414099219
- `random_source_test_layer` = 15

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 30 | -0.484 | +0.148 | **-0.632** | 0.0% | 0.0% | 0.0% |
| 14 | 30 | +2.593 | +0.129 | **+2.464** | 0.0% | 0.0% | 0.0% |
| 15 | 30 | +3.241 | -0.356 | **+3.597** | 0.0% | 0.0% | 0.0% |
| 16 | 30 | +1.265 | -0.403 | **+1.667** | 0.0% | 0.0% | 0.0% |
| 20 | 30 | +7.234 | -1.062 | **+8.297** | 0.0% | 0.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
