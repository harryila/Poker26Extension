# Decision-direction probe results

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`

## Probe (clean_check_or_call vs clean_legal_fold)
- Hidden dim: 4096
- Samples: 300 CHECK + 261 FOLD = 561 total
- Regularization (sklearn LR `C`): 0.01
- **5-fold CV accuracy: 0.998 ± 0.004**
- Per-fold: ['1.000', '1.000', '0.991', '1.000', '1.000']
- Train accuracy: 1.000
- ||w||₂: 0.2740
- cos(centroid_diff, w): **0.9542** (closer to 1 = probe weight aligns with mean-CHECK − mean-FOLD direction; this is a sanity check, not a separate finding)

## Projection of bucket residuals onto the learned direction
Sign convention: positive = CHECK side; negative = FOLD side. Units are pre-normalization residual / ||w||, so they are scale-comparable across buckets within this experiment.

| Bucket | n | mean projection | std |
|---|---:|---:|---:|
| clean_check_or_call | 300 | +24.714 | 3.805 |
| clean_legal_fold    | 261 | -23.155 | 2.780 |
| **illegal_fold**        | 24 | **-12.356** | 3.071 |

**Fraction of illegal_FOLDs on the FOLD side of midpoint: 100.0%** (midpoint = average of clean-bucket means)

## Reading guide
- High CV accuracy (>0.85) AND high cos(centroid, w) (>0.9): a single linear direction encodes the verb decision; the circuit is direction-projectable.
- illegal_FOLD mean projection well below clean_LF mean: the failure mode is *more* FOLD-aligned than legal FOLDs are. Consistent with §13's 'illegal_FOLDs lock in earlier and more confidently' finding.
- illegal_FOLD on FOLD side fraction ≥80%: failure mode lives on the same axis as the legal decision, just past the threshold.
