# SUPERSEDED — do not cite

This Ministral mode-balanced cell has too few matched-classified pairs to
fit a probe. **Do not include in the writeup.**

## What's wrong

After hand-matching CoT vs non-CoT logs by `(seed, decision_idx)` and then
classifying each side via `_reparse_one`, only **16 pairs** survive.
5-fold CV on n=16 → multiple folds with degenerate label distribution →
**CV accuracy = NaN ± NaN**. `cos(w_CoT, w_nonCoT) = +0.095` is
indistinguishable from zero given the tiny sample.

This is a Ministral data-distribution fact — Ministral generates very few
clean_check_or_call decisions in its non-CoT mode (also visible in the
context-stratified cell, where only FLOP and PREFLOP strata had ≥2
sources). Not a code bug; not fixable without re-inference at a different
pool.

## Replacement

For mode-balanced mode-stability claims, cite **Qwen only**:

```
results/mode_balanced_probe/qwen8b_l23/SUMMARY.md
   n=110 pairs, both modes own labels
   cos(w_CoT, w_nonCoT) = +0.51 (matched), centroid cos = +0.60
   CV accuracy: 100% / 100%
```

Llama (`results/mode_balanced_probe/llama8b_l14/SUMMARY.md`) uses
`label_source=cot` fallback (one mode had degenerate label distribution);
report with that caveat.

## Why we keep the directory

The 16-pair count itself is data — it's evidence that Ministral's non-CoT
behavior is bucket-skewed. The `summary.json` documents this. See
`AUDIT_FINDINGS.md` §"Mode-balanced probe" for full framing.
