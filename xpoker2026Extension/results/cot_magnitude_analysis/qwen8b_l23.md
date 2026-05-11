# CoT vs non-CoT residual magnitude analysis

- CoT probe:    `results/direction_probe/qwen8b_l23/raw_residuals.npz`
- non-CoT probe: `results/direction_probe_nocot/qwen8b_l23/raw_residuals.npz`

## Per-bucket residual L2 norms at L*

| Bucket | n | mean ||x|| | std | median |
|---|---:|---:|---:|---:|
| CoT  clean_CC | 300 | 158.86 | 1.53 | 159.24 |
| CoT  clean_LF | 261 | 157.33 | 1.47 | 157.24 |
| CoT  illegal_F | 24 | 156.79 | 0.78 | 156.94 |
| nonCoT clean_CC | 89 | 161.07 | 0.38 | 161.12 |
| nonCoT clean_LF | 100 | 164.84 | 0.97 | 165.15 |

## Centroid distance (mean(CHECK) − mean(FOLD), L2 norm)

- CoT:    50.17
- nonCoT: 87.22
- ratio (non-CoT / CoT): **1.74**

Reading: ratio < 1 means CoT residuals are MORE separated between CHECK and FOLD than non-CoT residuals (the verb decision is more distinctly encoded under CoT). Ratio > 1 means the opposite. A ratio near 1 means the geometry is preserved — the §18a 'attenuated prior' is visible at the *output discriminability* level (patches don't dominate) but not at the *centroid separation* level.

## Projection magnitudes onto each mode's own verb direction

| Mode | bucket | mean |proj| |
|---|---|---:|
| CoT | CHECK | 24.71 |
| CoT | FOLD  | 23.16 |
| nonCoT | CHECK | 43.17 |
| nonCoT | FOLD  | 43.96 |

## Reading guide
- **||x|| under non-CoT ≪ ||x|| under CoT**: the residual is compressed in non-CoT — small magnitude amplifies the relative weight of any prior bias.
- **Centroid distance smaller in non-CoT**: the verb decision is less distinctly encoded in non-CoT (patches struggle to express their content because the geometry is already biased toward the prior).
- **Projection magnitudes similar across modes**: the discriminating direction does similar work in both modes — the §18a finding is about the OUTPUT (action distribution softmax) not the residual representation itself.
