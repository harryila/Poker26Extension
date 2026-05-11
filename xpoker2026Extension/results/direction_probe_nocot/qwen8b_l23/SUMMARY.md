# Decision-direction probe results

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Enriched logs (pooled, n=1):
  - `logs/scaled_qwen8b_t0_s42_informative_v2_enriched.jsonl.gz`

## Probe (clean_check_or_call vs clean_legal_fold)
- Hidden dim: 4096
- Samples: 89 CHECK + 100 FOLD = 189 total
- Regularization (sklearn LR `C`): 0.01
- **5-fold CV accuracy: 1.000 ± 0.000**
- Per-fold: ['1.000', '1.000', '1.000', '1.000', '1.000']
- Train accuracy: 1.000
- ||w||₂: 0.1458
- cos(centroid_diff, w): **0.9989** (closer to 1 = probe weight aligns with mean-CHECK − mean-FOLD direction; this is a sanity check, not a separate finding)

## Projection of bucket residuals onto the learned direction
Sign convention: positive = CHECK side; negative = FOLD side. Units are pre-normalization residual / ||w||, so they are scale-comparable across buckets within this experiment.

| Bucket | n | mean projection | std |
|---|---:|---:|---:|
| clean_check_or_call | 89 | +43.167 | 1.727 |
| clean_legal_fold    | 100 | -43.961 | 1.260 |

## Reading guide
- High CV accuracy (>0.85) AND high cos(centroid, w) (>0.9): a single linear direction encodes the verb decision; the circuit is direction-projectable.
- illegal_FOLD mean projection well below clean_LF mean: the failure mode is *more* FOLD-aligned than legal FOLDs are. Consistent with §13's 'illegal_FOLDs lock in earlier and more confidently' finding.
- illegal_FOLD on FOLD side fraction ≥80%: failure mode lives on the same axis as the legal decision, just past the threshold.
