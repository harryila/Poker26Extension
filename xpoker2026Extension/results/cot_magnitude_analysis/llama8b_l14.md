# CoT vs non-CoT residual magnitude analysis

- CoT probe:    `results/direction_probe/llama8b_l14/raw_residuals.npz`
- non-CoT probe: `results/direction_probe_nocot/llama8b_l14/raw_residuals.npz`

## Per-bucket residual L2 norms at L*

| Bucket | n | mean ||x|| | std | median |
|---|---:|---:|---:|---:|
| CoT  clean_CC | 300 | 8.59 | 0.12 | 8.58 |
| CoT  clean_LF | 289 | 8.50 | 0.10 | 8.49 |
| CoT  illegal_F | 68 | 8.41 | 0.07 | 8.40 |
| nonCoT clean_CC | 272 | 8.41 | 0.18 | 8.29 |
| nonCoT clean_LF | 80 | 8.39 | 0.09 | 8.43 |

## Centroid distance (mean(CHECK) − mean(FOLD), L2 norm)

- CoT:    3.32
- nonCoT: 1.12
- ratio (non-CoT / CoT): **0.34**

Reading: ratio < 1 means CoT residuals are MORE separated between CHECK and FOLD than non-CoT residuals (the verb decision is more distinctly encoded under CoT). Ratio > 1 means the opposite. A ratio near 1 means the geometry is preserved — the §18a 'attenuated prior' is visible at the *output discriminability* level (patches don't dominate) but not at the *centroid separation* level.

## Projection magnitudes onto each mode's own verb direction

| Mode | bucket | mean |proj| |
|---|---|---:|
| CoT | CHECK | 1.46 |
| CoT | FOLD  | 1.86 |
| nonCoT | CHECK | 0.69 |
| nonCoT | FOLD  | 0.90 |

## Reading guide
- **||x|| under non-CoT ≪ ||x|| under CoT**: the residual is compressed in non-CoT — small magnitude amplifies the relative weight of any prior bias.
- **Centroid distance smaller in non-CoT**: the verb decision is less distinctly encoded in non-CoT (patches struggle to express their content because the geometry is already biased toward the prior).
- **Projection magnitudes similar across modes**: the discriminating direction does similar work in both modes — the §18a finding is about the OUTPUT (action distribution softmax) not the residual representation itself.
