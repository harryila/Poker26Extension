# Decision-direction probe results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Enriched logs (pooled, n=1):
  - `logs/scaled_llama8b_t0_s42_informative_v2_enriched.jsonl.gz`

## Probe (clean_check_or_call vs clean_legal_fold)
- Hidden dim: 4096
- Samples: 272 CHECK + 80 FOLD = 352 total
- Regularization (sklearn LR `C`): 0.01
- **5-fold CV accuracy: 0.773 ± 0.002**
- Per-fold: ['0.775', '0.775', '0.771', '0.771', '0.771']
- Train accuracy: 0.773
- ||w||₂: 0.5136
- cos(centroid_diff, w): **0.9917** (closer to 1 = probe weight aligns with mean-CHECK − mean-FOLD direction; this is a sanity check, not a separate finding)

## Projection of bucket residuals onto the learned direction
Sign convention: positive = CHECK side; negative = FOLD side. Units are pre-normalization residual / ||w||, so they are scale-comparable across buckets within this experiment.

| Bucket | n | mean projection | std |
|---|---:|---:|---:|
| clean_check_or_call | 272 | +0.556 | 0.450 |
| clean_legal_fold    | 80 | -0.555 | 0.736 |

## Reading guide
- High CV accuracy (>0.85) AND high cos(centroid, w) (>0.9): a single linear direction encodes the verb decision; the circuit is direction-projectable.
- illegal_FOLD mean projection well below clean_LF mean: the failure mode is *more* FOLD-aligned than legal FOLDs are. Consistent with §13's 'illegal_FOLDs lock in earlier and more confidently' finding.
- illegal_FOLD on FOLD side fraction ≥80%: failure mode lives on the same axis as the legal decision, just past the threshold.
