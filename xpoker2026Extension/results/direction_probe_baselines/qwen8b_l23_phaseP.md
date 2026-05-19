# Direction-probe baselines

- Probe: `results/direction_probe/qwen8b_l23/raw_residuals.npz`
- Samples: 600, hidden_dim: 4096
- Class balancing: `upsampled`
- Cross-task feature: `position`

## Probe accuracy comparison

| Probe | CV accuracy | Notes |
|---|---:|---|
| **Learned probe** (verb labels) | **0.998 ± 0.003** | the actual probe |
| Permuted-label control (20 shuffles) | 0.498 ± 0.027 | shuffled labels — chance (~0.50) expected |
| Random-direction (BEST threshold, 20 trials) | 0.789 ± 0.134 | upper bound: oracle threshold per trial |
| Random-direction (FIXED median threshold, 20 trials) | 0.743 ± 0.149 | conservative: no per-trial threshold optimization |

## Reading guide

- **Learned ≫ permuted-label**: confirms the probe is learning from the residual, not memorizing the labels.
- **Learned ≫ random-direction**: confirms a *specific* direction encodes the verb decision; not just any 1-D projection works.
- **Cross-task accuracy notable**: the residuals also encode other situational features. If cross-task accuracy is also high, it doesn't *contradict* the verb-direction finding — residuals encode many things — but it shows the residuals are information-rich at this layer.
- **Cross-task accuracy near chance**: confirms the verb direction is task-specific, not generic high-info.
