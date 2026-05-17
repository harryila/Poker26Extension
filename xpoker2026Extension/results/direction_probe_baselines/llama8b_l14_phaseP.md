# Direction-probe baselines

- Probe: `results/direction_probe/llama8b_l14/raw_residuals.npz`
- Samples: 600, hidden_dim: 4096
- Class balancing: `upsampled`
- Cross-task feature: `position`

## Probe accuracy comparison

| Probe | CV accuracy | Notes |
|---|---:|---|
| **Learned probe** (verb labels) | **0.988 ± 0.008** | the actual probe |
| Permuted-label control | 0.497 ± 0.026 | shuffled labels — chance (~0.50) expected |
| Random-direction (best threshold, 20 trials) | 0.759 ± 0.115 | 1-D classification on a random axis |

## Reading guide

- **Learned ≫ permuted-label**: confirms the probe is learning from the residual, not memorizing the labels.
- **Learned ≫ random-direction**: confirms a *specific* direction encodes the verb decision; not just any 1-D projection works.
- **Cross-task accuracy notable**: the residuals also encode other situational features. If cross-task accuracy is also high, it doesn't *contradict* the verb-direction finding — residuals encode many things — but it shows the residuals are information-rich at this layer.
- **Cross-task accuracy near chance**: confirms the verb direction is task-specific, not generic high-info.
