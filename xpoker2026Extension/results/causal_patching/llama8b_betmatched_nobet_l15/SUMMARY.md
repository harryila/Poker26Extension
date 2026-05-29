# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layers: [15]

## Controls
- `baseline_top1_match_rate` = 0.9666666666666667
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'15': {'mean_delta': -0.8821379982376663, 'std_delta': 1.4906155937637746, 'n': 5, 'n_random_sources': 5, 'n_random_targets': 1}}
- `random_source_n` = 5
- `random_source_mean_delta` = -0.8821379982376663
- `random_source_test_layer` = 15

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 300 | +10.296 | -0.882 | **+11.178** | 93.3% | 0.0% | 6.7% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
