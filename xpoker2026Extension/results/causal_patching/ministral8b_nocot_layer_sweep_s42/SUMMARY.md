# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched log: `logs/scaled_ministral8b_t0_s42_informative_v2_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `clean_legal_fold` (n=30)
- Layers: [8, 10, 12, 14, 16, 18, 20]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'8': {'mean_delta': 0.0038376446296553013, 'n': 5}, '10': {'mean_delta': -0.018628388355253378, 'n': 5}, '12': {'mean_delta': -0.04190847915253215, 'n': 5}, '14': {'mean_delta': 0.034209033981756676, 'n': 5}, '16': {'mean_delta': 0.2544652682354638, 'n': 5}, '18': {'mean_delta': 0.36768346948901326, 'n': 5}, '20': {'mean_delta': 0.3677328353959609, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 0.034209033981756676
- `random_source_test_layer` = 14

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 300 | +0.041 | +0.004 | **+0.037** | 0.0% | 100.0% | 0.0% |
| 10 | 300 | -0.035 | -0.019 | **-0.016** | 0.0% | 100.0% | 0.0% |
| 12 | 300 | -0.139 | -0.042 | **-0.097** | 0.0% | 100.0% | 0.0% |
| 14 | 300 | +0.021 | +0.034 | **-0.013** | 0.0% | 100.0% | 0.0% |
| 16 | 300 | +0.930 | +0.254 | **+0.675** | 13.3% | 28.0% | 58.7% |
| 18 | 300 | +1.603 | +0.368 | **+1.236** | 40.0% | 0.0% | 60.0% |
| 20 | 300 | +1.637 | +0.368 | **+1.269** | 40.0% | 0.0% | 60.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
