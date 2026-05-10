# Decision-direction probe results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`

## Probe (clean_check_or_call vs clean_legal_fold)
- Hidden dim: 4096
- Samples: 300 CHECK + 289 FOLD = 589 total
- Regularization (sklearn LR `C`): 0.01
- **5-fold CV accuracy: 0.988 ± 0.009**
- Per-fold: ['0.992', '0.983', '0.992', '1.000', '0.974']
- Train accuracy: 0.988
- ||w||₂: 1.1544
- cos(centroid_diff, w): **0.9935** (closer to 1 = probe weight aligns with mean-CHECK − mean-FOLD direction; this is a sanity check, not a separate finding)

## Projection of bucket residuals onto the learned direction
Sign convention: positive = CHECK side; negative = FOLD side. Units are pre-normalization residual / ||w||, so they are scale-comparable across buckets within this experiment.

| Bucket | n | mean projection | std |
|---|---:|---:|---:|
| clean_check_or_call | 300 | +1.438 | 0.536 |
| clean_legal_fold    | 289 | -1.860 | 0.395 |
| **illegal_fold**        | 68 | **-1.270** | 0.356 |

**Fraction of illegal_FOLDs on the FOLD side of midpoint: 98.5%** (midpoint = average of clean-bucket means)

## Reading guide
- High CV accuracy (>0.85) AND high cos(centroid, w) (>0.9): a single linear direction encodes the verb decision; the circuit is direction-projectable.
- illegal_FOLD mean projection well below clean_LF mean: the failure mode is *more* FOLD-aligned than legal FOLDs are. Consistent with §13's 'illegal_FOLDs lock in earlier and more confidently' finding.
- illegal_FOLD on FOLD side fraction ≥80%: failure mode lives on the same axis as the legal decision, just past the threshold.
