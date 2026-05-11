# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched log: `logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `clean_legal_fold` (n=30)
- Layers: [15, 18, 21, 23, 25, 28, 31]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'15': {'mean_delta': -0.04877934562495625, 'n': 5}, '18': {'mean_delta': 0.4514668501642703, 'n': 5}, '21': {'mean_delta': 2.0993268781386347, 'n': 5}, '23': {'mean_delta': 3.3998533036733236, 'n': 5}, '25': {'mean_delta': 3.6505294809663638, 'n': 5}, '28': {'mean_delta': 3.650669889647472, 'n': 5}, '31': {'mean_delta': 4.350123711280406, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 3.3998533036733236
- `random_source_test_layer` = 23

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 300 | -0.361 | -0.049 | **-0.312** | 0.0% | 100.0% | 0.0% |
| 18 | 300 | +2.035 | +0.451 | **+1.584** | 0.0% | 100.0% | 0.0% |
| 21 | 300 | +12.280 | +2.099 | **+10.180** | 7.0% | 0.0% | 93.0% |
| 23 | 300 | +23.610 | +3.400 | **+20.211** | 88.7% | 0.0% | 11.3% |
| 25 | 300 | +24.490 | +3.651 | **+20.839** | 80.0% | 0.0% | 20.0% |
| 28 | 300 | +24.755 | +3.651 | **+21.105** | 90.0% | 0.0% | 10.0% |
| 31 | 300 | +27.687 | +4.350 | **+23.337** | 100.0% | 0.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
