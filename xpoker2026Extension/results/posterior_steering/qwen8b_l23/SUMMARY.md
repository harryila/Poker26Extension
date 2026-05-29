# Posterior steering — does the calibration direction de-bias the decision?

- model=Qwen/Qwen3-8B layer=23 target=clean_legal_fold n=60
- alpha in units of mean residual norm; `trash_direction` vs `random_control`.

| alpha | trash: mean ΔCHECK−FOLD | trash: top-1→CHECK | control: mean Δ | control: top-1→CHECK |
|---:|---:|---:|---:|---:|
| 0.0 | -22.04 | 0% | -22.04 | 0% |
| 2.0 | -23.06 | 0% | -24.48 | 0% |
| 4.0 | -11.43 | 0% | -12.86 | 0% |
| 8.0 | -6.70 | 0% | -3.18 | 0% |

## Reading
- A monotone CHECK−FOLD increase with alpha for `trash_direction` that EXCEEDS the `random_control` = the calibration direction causally de-biases the over-fold tendency (controllable steering). If trash≈control, the effect is generic norm perturbation, not the direction.
