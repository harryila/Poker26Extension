# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- **Zero-ablation mode**: source residual is the all-zeros tensor (no real source sampled). The `--source-bucket` value below is used only to exclude that bucket from the random-null pool in control 3.
- Source bucket: `clean_check_or_call` (n=1)
- Target bucket: `illegal_fold` (n=30)
- Layers: [12, 13, 14, 15, 18]

## Controls
- `baseline_top1_match_rate` = 0.9666666666666667
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'12': {'mean_delta': 0.16600311900679543, 'n': 5}, '13': {'mean_delta': 0.8121337649388757, 'n': 5}, '14': {'mean_delta': 1.5600179825046099, 'n': 5}, '15': {'mean_delta': 2.6602523871297796, 'n': 5}, '18': {'mean_delta': 3.1866824591928085, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 1.5600179825046099
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 30 | +5.862 | +0.166 | **+5.696** | 0.0% | 0.0% | 0.0% |
| 13 | 30 | +5.339 | +0.812 | **+4.527** | 20.0% | 10.0% | 0.0% |
| 14 | 30 | +3.020 | +1.560 | **+1.460** | 0.0% | 0.0% | 0.0% |
| 15 | 30 | +9.079 | +2.660 | **+6.419** | 0.0% | 0.0% | 0.0% |
| 18 | 30 | +3.216 | +3.187 | **+0.029** | 0.0% | 0.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
