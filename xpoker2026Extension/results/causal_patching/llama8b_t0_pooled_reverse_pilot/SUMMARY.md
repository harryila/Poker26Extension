# Causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_legal_fold` (n=10)
- Target bucket: `clean_check_or_call` (n=30)
- Layers: [10, 14, 18, 24, 30]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'10': {'mean_delta': 0.07516838546626303, 'n': 5}, '14': {'mean_delta': 0.1852927472043479, 'n': 5}, '18': {'mean_delta': -1.1766742143017055, 'n': 5}, '24': {'mean_delta': -1.8633781768111297, 'n': 5}, '30': {'mean_delta': -1.9503390326777073, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = -1.1766742143017055
- `random_source_test_layer` = 18

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 300 | -0.099 | +0.075 | **-0.175** | 100.0% | 0.0% |
| 14 | 300 | -10.144 | +0.185 | **-10.330** | 40.0% | 60.0% |
| 18 | 300 | -15.585 | -1.177 | **-14.409** | 0.0% | 100.0% |
| 24 | 300 | -16.738 | -1.863 | **-14.875** | 0.0% | 100.0% |
| 30 | 300 | -16.802 | -1.950 | **-14.852** | 0.0% | 100.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation.
