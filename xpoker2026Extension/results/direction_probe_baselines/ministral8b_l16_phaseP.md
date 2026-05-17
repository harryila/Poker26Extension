# Direction-probe baselines

- Probe: `results/direction_probe/ministral8b_l16/raw_residuals.npz`
- Samples: 596, hidden_dim: 4096
- Class balancing: `upsampled`
- Cross-task feature: `position`

## Probe accuracy comparison

| Probe | CV accuracy | Notes |
|---|---:|---|
| **Learned probe** (verb labels) | **1.000 ± 0.000** | the actual probe |
| Permuted-label control | 0.478 ± 0.013 | shuffled labels — chance (~0.50) expected |
| Random-direction (best threshold, 20 trials) | 0.883 ± 0.102 | 1-D classification on a random axis |

## Reading guide

- **Learned ≫ permuted-label**: confirms the probe is learning from the residual, not memorizing the labels.
- **Learned ≫ random-direction**: confirms a *specific* direction encodes the verb decision; not just any 1-D projection works.
- **Cross-task accuracy notable**: the residuals also encode other situational features. If cross-task accuracy is also high, it doesn't *contradict* the verb-direction finding — residuals encode many things — but it shows the residuals are information-rich at this layer.
- **Cross-task accuracy near chance**: confirms the verb direction is task-specific, not generic high-info.
