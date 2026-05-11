# Position-sweep direction projection

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Direction probe weights: `results/direction_probe/qwen8b_l23/raw_residuals.npz`
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample cap: 50

Sign convention: positive = CHECK side (signed projection onto the probe's weight vector, normalized by ||w||). Position is RELATIVE to the verb-emission position (0 = the last input position whose lm_head output predicts the verb).

## Mean projection at each relative position

| rel_pos | `clean_check_or_call` mean ± std (n) | `clean_legal_fold` mean ± std (n) | `illegal_fold` mean ± std (n) |
|---:|---|---|---|
| -300 | +46.45 ± 219.41 (33) | +31.88 ± 179.41 (50) | +6.00 ± 2.55 (23) |
| -200 | +5.08 ± 6.85 (50) | +6.68 ± 4.61 (50) | +4.12 ± 6.59 (24) |
| -100 | +5.02 ± 4.84 (50) | +3.81 ± 4.95 (50) | +4.10 ± 4.79 (24) |
| -50 | +12.68 ± 6.13 (50) | +10.72 ± 4.17 (50) | +10.60 ± 2.97 (24) |
| -20 | +16.33 ± 7.11 (50) | +10.07 ± 4.98 (50) | +11.22 ± 5.28 (24) |
| -10 | +13.27 ± 7.07 (50) | -5.13 ± 2.92 (50) | -3.79 ± 3.37 (24) |
| -5 | -2.34 ± 0.40 (50) | -4.35 ± 0.25 (50) | -4.18 ± 0.27 (24) |
| -2 | +16.63 ± 2.26 (50) | -12.25 ± 2.16 (50) | -7.43 ± 2.25 (24) |
| -1 | +19.15 ± 3.55 (50) | -11.39 ± 1.84 (50) | -7.31 ± 2.39 (24) |
| +0 | +24.75 ± 4.63 (50) | -22.61 ± 3.34 (50) | -12.36 ± 3.07 (24) |
| +1 | +13.89 ± 1.52 (50) | -17.92 ± 2.35 (50) | -13.33 ± 3.37 (24) |
| +2 | +13.02 ± 1.79 (50) | -11.99 ± 0.60 (50) | -11.20 ± 0.82 (24) |
| +5 | — | — | — |
| +10 | — | — | — |
| +20 | — | — | — |
| +50 | — | — | — |
| +100 | — | — | — |

## Reading guide

- **`rel_pos = 0` is the verb-emission position** — by construction this is where the patching experiments measure the decision. The mean projection here should match the per-bucket projections from the direction probe (within sampling noise).
- **`rel_pos < 0` (earlier in response)**: where in the response trace does the decision crystallize? Compute-then-commit predicts: projection should be near zero (undecided) for very negative offsets (early in the response, still reasoning), then diverge sharply approaching the verb position.
- **`rel_pos > 0` (after the verb)**: how does the residual evolve after the model has emitted the verb? If the projection stays at the bucket's verb-position level, the decision is consistently encoded throughout. If it relaxes back toward zero, the model's residual moves on quickly.
- **Cross-bucket comparison at very negative offsets**: if the buckets are INDISTINGUISHABLE there, the prompt itself doesn't pre-determine the verb (no prompt-level shortcut). If they ARE distinguishable, the prompt context alone partially predicts the decision (and the residual is doing less work than we thought).
- **Variance comparison**: `illegal_fold` is expected to have higher variance at the verb position (consistent with §17a's higher attention entropy).
