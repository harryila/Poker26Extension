# Direction-probe baselines

- Probe: `results/direction_probe/qwen8b_l23/raw_residuals.npz`
- Samples: 561, hidden_dim: 4096

## Probe accuracy comparison

| Probe | CV accuracy | Notes |
|---|---:|---|
| **Learned probe** (verb labels) | **0.998 ± 0.004** | the actual probe |
| Permuted-label control | 0.508 ± 0.042 | shuffled labels — chance (~0.50) expected |
| Random-direction (best threshold, 20 trials) | 0.775 ± 0.128 | 1-D classification on a random axis |
| Cross-task (`bet_to_call > 0`) | 1.000 ± 0.000 | different label, same residuals (n_pos=363, n_neg=198) |

## Reading guide

- **Learned ≫ permuted-label**: confirms the probe is learning from the residual, not memorizing the labels.
- **Learned ≫ random-direction**: confirms a *specific* direction encodes the verb decision; not just any 1-D projection works.
- **Cross-task accuracy notable**: the residuals also encode other situational features. If cross-task accuracy is also high, it doesn't *contradict* the verb-direction finding — residuals encode many things — but it shows the residuals are information-rich at this layer.
- **Cross-task accuracy near chance**: confirms the verb direction is task-specific, not generic high-info.
