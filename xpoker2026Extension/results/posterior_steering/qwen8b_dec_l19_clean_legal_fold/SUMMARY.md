# Posterior steering — does the calibration direction de-bias the decision?

- model=Qwen/Qwen3-8B layer=19 target=clean_legal_fold n=60
- alpha in units of mean residual norm; `trash_direction` vs `random_control`.

| alpha | trash: mean ΔCHECK−FOLD | trash: top-1→CHECK | control: mean Δ | control: top-1→CHECK |
|---:|---:|---:|---:|---:|
| -8.0 | +3.00 | 0% | -4.13 | 0% |
| -4.0 | -0.29 | 0% | -1.17 | 0% |
| 0.0 | -22.04 | 0% | -22.04 | 0% |
| 4.0 | -2.69 | 0% | -10.07 | 0% |
| 8.0 | -5.41 | 0% | -1.85 | 0% |

## Reading
- A monotone CHECK−FOLD increase with alpha for `trash_direction` that EXCEEDS the `random_control` = the calibration direction causally de-biases the over-fold tendency (controllable steering). If trash≈control, the effect is generic norm perturbation, not the direction.
