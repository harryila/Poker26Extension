# Posterior steering — does the calibration direction de-bias the decision?

- model=Qwen/Qwen3-8B layer=23 target=illegal_fold n=24
- alpha in units of mean residual norm; `trash_direction` vs `random_control`.

| alpha | trash: mean ΔCHECK−FOLD | trash: top-1→CHECK | control: mean Δ | control: top-1→CHECK |
|---:|---:|---:|---:|---:|
| -8.0 | +3.02 | 0% | -2.71 | 0% |
| -4.0 | -0.79 | 0% | -3.13 | 0% |
| 0.0 | -11.81 | 0% | -11.81 | 0% |
| 4.0 | -6.80 | 0% | -11.06 | 0% |
| 8.0 | -4.65 | 0% | -3.22 | 0% |

## Reading
- A monotone CHECK−FOLD increase with alpha for `trash_direction` that EXCEEDS the `random_control` = the calibration direction causally de-biases the over-fold tendency (controllable steering). If trash≈control, the effect is generic norm perturbation, not the direction.
