# Direction probe comparison: `llama L=14 CoT` vs `llama L=14 non-CoT`

- Probe A: `results/direction_probe/llama8b_l14/raw_residuals.npz` (llama L=14 CoT)
- Probe B: `results/direction_probe_nocot/llama8b_l14/raw_residuals.npz` (llama L=14 non-CoT)
- Hidden dim: 4096

## Direct cosines (sign matters: positive = same axis, negative = opposite axis, near zero = orthogonal)

- cos(w_A, w_B) = **+0.2698**  (probe weight vectors)
- cos(centroid_diff_A, centroid_diff_B) = **+0.2218**  (mean-CHECK − mean-FOLD axes)
- cos(w_A, centroid_diff_B) = **+0.2346**
- cos(w_B, centroid_diff_A) = **+0.2540**

## Cross-projection (how well does each direction discriminate the OTHER set?)

Mean projection difference (CHECK mean − FOLD mean) on the OTHER probe's residuals using THIS probe's weight vector. Positive = direction discriminates correctly.

- B residuals projected onto w_A: **+0.263** (✅ correct sign)
- A residuals projected onto w_B: **+0.843** (✅ correct sign)

## Reading guide

- **cos(w_A, w_B) > 0.85**: probes recover the SAME direction in residual space. The verb-decision direction is shared between the two conditions (e.g. CoT vs non-CoT). This is the strongest possible 'shared circuit' evidence.
- **0.5 < cos(w_A, w_B) < 0.85**: directions are correlated but not identical. The decisions are encoded along similar (but not the same) axis. Could reflect overlapping circuits or shared features with condition-specific extras.
- **cos(w_A, w_B) < 0.5**: directions are largely independent. Each condition uses a different axis to encode the verb. This would be evidence AGAINST a shared circuit.
- **Cross-projection signs**: if both signs are correct (CHECK > FOLD when projected onto the other direction), the decision axes are functionally interchangeable. If signs are wrong, the directions are unrelated or anti-correlated.
