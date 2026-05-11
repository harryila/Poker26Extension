# Position-sweep direction projection

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Direction probe weights: `results/direction_probe/ministral8b_l16/raw_residuals.npz`
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample cap: 50

Sign convention: positive = CHECK side (signed projection onto the probe's weight vector, normalized by ||w||). Position is RELATIVE to the verb-emission position (0 = the last input position whose lm_head output predicts the verb).

## Mean projection at each relative position

| rel_pos | `clean_check_or_call` mean ± std (n) | `clean_legal_fold` mean ± std (n) | `illegal_fold` mean ± std (n) |
|---:|---|---|---|
| -300 | +0.13 ± 0.12 (28) | +0.36 ± 1.42 (50) | +1.38 ± 2.88 (15) |
| -200 | -0.11 ± 0.51 (33) | -0.34 ± 0.63 (50) | +0.06 ± 0.28 (50) |
| -100 | +0.02 ± 0.30 (33) | +0.19 ± 0.32 (50) | +0.34 ± 0.27 (50) |
| -50 | +0.28 ± 0.20 (33) | +0.31 ± 0.18 (50) | +0.25 ± 0.27 (50) |
| -20 | +0.35 ± 0.34 (33) | +0.09 ± 0.21 (50) | +0.21 ± 0.19 (50) |
| -10 | +0.53 ± 0.31 (33) | -0.50 ± 0.50 (50) | -0.22 ± 0.30 (50) |
| -5 | -0.56 ± 0.03 (33) | -0.76 ± 0.02 (50) | -0.74 ± 0.02 (50) |
| -2 | -0.27 ± 0.13 (33) | -1.46 ± 0.04 (50) | -1.36 ± 0.05 (50) |
| -1 | +0.56 ± 0.24 (33) | -1.74 ± 0.09 (50) | -1.48 ± 0.18 (50) |
| +0 | +1.14 ± 0.32 (33) | -2.01 ± 0.10 (50) | -1.50 ± 0.23 (50) |
| +1 | +0.20 ± 0.06 (33) | +0.25 ± 0.03 (50) | +0.22 ± 0.13 (50) |
| +2 | +0.96 ± 0.23 (33) | -0.85 ± 0.03 (50) | -0.81 ± 0.02 (50) |
| +5 | +0.50 ± 0.10 (13) | — | — |
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
