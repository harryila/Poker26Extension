# Causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched log: `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=4)
- Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_per_layer` = {'0': {'mean_delta': 0.14999510604502575, 'n': 5}, '1': {'mean_delta': 0.04999812511988751, 'n': 5}, '2': {'mean_delta': -1.3018529784858402e-06, 'n': 5}, '3': {'mean_delta': -0.05000122346216358, 'n': 5}, '4': {'mean_delta': 0.04999748233692429, 'n': 5}, '5': {'mean_delta': 0.09999504124981229, 'n': 5}, '6': {'mean_delta': 0.04999747774078287, 'n': 5}, '7': {'mean_delta': 0.04999748230701186, 'n': 5}, '8': {'mean_delta': 0.09999575314035525, 'n': 5}, '9': {'mean_delta': -2.5215493650421193e-06, 'n': 5}, '10': {'mean_delta': -6.474612717966011e-07, 'n': 5}, '11': {'mean_delta': 0.19999538648642384, 'n': 5}, '12': {'mean_delta': 0.19999374163337932, 'n': 5}, '13': {'mean_delta': 0.499992963035902, 'n': 5}, '14': {'mean_delta': 0.64999353417613, 'n': 5}, '15': {'mean_delta': 0.799994015101251, 'n': 5}, '16': {'mean_delta': 0.6999948646263704, 'n': 5}, '17': {'mean_delta': 3.511426612590185e-06, 'n': 5}, '18': {'mean_delta': 0.5499995568679978, 'n': 5}, '19': {'mean_delta': 1.4999918672282333, 'n': 5}, '20': {'mean_delta': 2.400000659460768, 'n': 5}, '21': {'mean_delta': 2.30004883692681, 'n': 5}, '22': {'mean_delta': 3.450269803191843, 'n': 5}, '23': {'mean_delta': 3.650401739183797, 'n': 5}, '24': {'mean_delta': 2.10137652279971, 'n': 5}, '25': {'mean_delta': 2.626474085530795, 'n': 5}, '26': {'mean_delta': 2.351696580507462, 'n': 5}, '27': {'mean_delta': 1.7765341754013817, 'n': 5}, '28': {'mean_delta': 1.152993173649876, 'n': 5}, '29': {'mean_delta': 1.7015232783412089, 'n': 5}, '30': {'mean_delta': 1.6521310694132751, 'n': 5}, '31': {'mean_delta': 1.7513153253900626, 'n': 5}, '32': {'mean_delta': 1.5511334448327894, 'n': 5}, '33': {'mean_delta': 1.2761206208134794, 'n': 5}, '34': {'mean_delta': 0.9014854461091197, 'n': 5}, '35': {'mean_delta': 0.8768866445915414, 'n': 5}}
- `random_source_n` = 5
- `random_source_mean_delta` = 0.5499995568679978
- `random_source_test_layer` = 18

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | random null Δ | specificity-adjusted | top-1 → CHECK-family | top-1 → FOLD-family | top-1 → BET_RAISE-family |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40 | -0.050 | +0.150 | **-0.200** | 0.0% | 100.0% | 0.0% |
| 1 | 40 | -0.013 | +0.050 | **-0.062** | 0.0% | 100.0% | 0.0% |
| 2 | 40 | -0.031 | -0.000 | **-0.031** | 0.0% | 100.0% | 0.0% |
| 3 | 40 | -0.025 | -0.050 | **+0.025** | 0.0% | 100.0% | 0.0% |
| 4 | 40 | -0.031 | +0.050 | **-0.081** | 0.0% | 100.0% | 0.0% |
| 5 | 40 | -0.037 | +0.100 | **-0.137** | 0.0% | 100.0% | 0.0% |
| 6 | 40 | -0.019 | +0.050 | **-0.069** | 0.0% | 100.0% | 0.0% |
| 7 | 40 | -0.056 | +0.050 | **-0.106** | 0.0% | 100.0% | 0.0% |
| 8 | 40 | -0.044 | +0.100 | **-0.144** | 0.0% | 100.0% | 0.0% |
| 9 | 40 | -0.112 | -0.000 | **-0.112** | 0.0% | 100.0% | 0.0% |
| 10 | 40 | -0.137 | -0.000 | **-0.137** | 0.0% | 100.0% | 0.0% |
| 11 | 40 | -0.119 | +0.200 | **-0.319** | 0.0% | 100.0% | 0.0% |
| 12 | 40 | -0.050 | +0.200 | **-0.250** | 0.0% | 100.0% | 0.0% |
| 13 | 40 | +0.112 | +0.500 | **-0.387** | 0.0% | 100.0% | 0.0% |
| 14 | 40 | +0.312 | +0.650 | **-0.338** | 0.0% | 100.0% | 0.0% |
| 15 | 40 | +0.319 | +0.800 | **-0.481** | 0.0% | 100.0% | 0.0% |
| 16 | 40 | +0.319 | +0.700 | **-0.381** | 0.0% | 100.0% | 0.0% |
| 17 | 40 | -0.456 | +0.000 | **-0.456** | 0.0% | 100.0% | 0.0% |
| 18 | 40 | +0.175 | +0.550 | **-0.375** | 0.0% | 100.0% | 0.0% |
| 19 | 40 | +2.937 | +1.500 | **+1.437** | 0.0% | 100.0% | 0.0% |
| 20 | 40 | +6.087 | +2.400 | **+3.687** | 7.5% | 92.5% | 0.0% |
| 21 | 40 | +11.600 | +2.300 | **+9.300** | 35.0% | 65.0% | 0.0% |
| 22 | 40 | +20.444 | +3.450 | **+16.993** | 90.0% | 10.0% | 0.0% |
| 23 | 40 | +28.700 | +3.650 | **+25.050** | 100.0% | 0.0% | 0.0% |
| 24 | 40 | +29.144 | +2.101 | **+27.042** | 100.0% | 0.0% | 0.0% |
| 25 | 40 | +31.737 | +2.626 | **+29.111** | 100.0% | 0.0% | 0.0% |
| 26 | 40 | +32.419 | +2.352 | **+30.067** | 100.0% | 0.0% | 0.0% |
| 27 | 40 | +33.159 | +1.777 | **+31.383** | 100.0% | 0.0% | 0.0% |
| 28 | 40 | +33.506 | +1.153 | **+32.353** | 100.0% | 0.0% | 0.0% |
| 29 | 40 | +34.469 | +1.702 | **+32.767** | 100.0% | 0.0% | 0.0% |
| 30 | 40 | +36.662 | +1.652 | **+35.010** | 100.0% | 0.0% | 0.0% |
| 31 | 40 | +36.606 | +1.751 | **+34.855** | 100.0% | 0.0% | 0.0% |
| 32 | 40 | +36.406 | +1.551 | **+34.855** | 100.0% | 0.0% | 0.0% |
| 33 | 40 | +36.062 | +1.276 | **+34.786** | 100.0% | 0.0% | 0.0% |
| 34 | 40 | +36.228 | +0.901 | **+35.327** | 100.0% | 0.0% | 0.0% |
| 35 | 40 | +37.850 | +0.877 | **+36.973** | 100.0% | 0.0% | 0.0% |

**Specificity-adjusted Δ** = (source effect) − (random alt-bucket source effect at the same layer). This is the writeup-ready signal: it isolates the contribution of the source's content from any generic 'patching at layer L breaks the model' effect. Sign convention: positive Δ = patched residual pushes the model toward CHECK_CALL, negative Δ = pushes toward FOLD. For FORWARD runs (CHECK source → FOLD-committed target) read the `top-1 → CHECK-family` column. For REVERSE runs (FOLD source → CHECK-committed target) read the `top-1 → FOLD-family` column AND expect specificity-adjusted Δ to go NEGATIVE at saturation. For VERB-GENERALITY runs (BET_RAISE source → CHECK target) read the `top-1 → BET_RAISE-family` column AND note that the headline Δ here is BET_RAISE-vs-CHECK, NOT CHECK-vs-FOLD — the printed `mean Δlogit(CHECK − FOLD)` column is then a sanity check (close to zero means the patch is verb-specific rather than verb-generic).
