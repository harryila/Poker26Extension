# Position-sweep direction projection

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Direction probe weights: `results/direction_probe/llama8b_l14/raw_residuals.npz`
- Buckets: ['clean_check_or_call', 'clean_legal_fold', 'illegal_fold']
- Per-bucket sample cap: 50

Sign convention: positive = CHECK side (signed projection onto the probe's weight vector, normalized by ||w||). Position is RELATIVE to the verb-emission position (0 = the last input position whose lm_head output predicts the verb).

## Mean projection at each relative position

| rel_pos | `clean_check_or_call` mean ± std (n) | `clean_legal_fold` mean ± std (n) | `illegal_fold` mean ± std (n) |
|---:|---|---|---|
| -300 | -0.01 ± 0.26 (50) | +0.01 ± 0.22 (50) | +0.06 ± 0.15 (50) |
| -200 | -0.13 ± 0.43 (50) | -0.10 ± 0.26 (50) | -0.11 ± 0.30 (50) |
| -100 | +0.00 ± 0.23 (50) | -0.12 ± 0.17 (50) | -0.03 ± 0.21 (50) |
| -50 | -0.08 ± 0.29 (50) | -0.22 ± 0.18 (50) | -0.14 ± 0.23 (50) |
| -20 | +0.13 ± 0.30 (50) | -0.33 ± 0.41 (50) | -0.41 ± 0.30 (50) |
| -10 | +0.04 ± 0.33 (50) | -1.06 ± 0.41 (50) | -1.00 ± 0.34 (50) |
| -5 | -0.08 ± 0.05 (50) | -0.23 ± 0.02 (50) | -0.22 ± 0.02 (50) |
| -2 | +0.59 ± 0.39 (50) | -1.49 ± 0.21 (50) | -1.16 ± 0.20 (50) |
| -1 | +0.74 ± 0.25 (50) | -1.06 ± 0.40 (50) | -0.82 ± 0.33 (50) |
| +0 | +1.45 ± 0.53 (50) | -1.99 ± 0.36 (50) | -1.28 ± 0.36 (50) |
| +1 | +1.17 ± 0.27 (50) | -0.99 ± 0.44 (50) | -0.68 ± 0.27 (50) |
| +2 | +1.00 ± 0.24 (50) | -1.63 ± 0.10 (50) | -1.60 ± 0.07 (50) |
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
