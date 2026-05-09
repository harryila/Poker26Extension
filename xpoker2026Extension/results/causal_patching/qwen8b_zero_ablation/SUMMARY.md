# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- **Zero-ablation mode**: source residual is the all-zeros tensor (no real source sampled). The `--source-bucket` value below is used only to exclude that bucket from the random-null pool in control 3.
- Source bucket: `clean_check_or_call` (n=1)
- Target bucket: `illegal_fold` (n=24)
- Layers: [18, 20, 22, 24, 30]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'18': {'mean_delta': 0.8000004972430304, 'n': 5}, '20': {'mean_delta': 5.000000849260763, 'n': 5}, '22': {'mean_delta': 12.950097008823278, 'n': 5}, '24': {'mean_delta': 14.550319645883942, 'n': 5}, '30': {'mean_delta': 11.93142485240011, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 12.950097008823278
- `random_source_test_layer` = 22

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 24 | +7.764 | +0.800 | **+6.964** | 0.0% | 0.0% | 0.0% |
| 20 | 24 | +7.221 | +5.000 | **+2.221** | 0.0% | 0.0% | 0.0% |
| 22 | 24 | +9.715 | +12.950 | **-3.235** | 0.0% | 0.0% | 0.0% |
| 24 | 24 | +6.885 | +14.550 | **-7.666** | 0.0% | 4.2% | 0.0% |
| 30 | 24 | +6.520 | +11.931 | **-5.411** | 0.0% | 0.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
