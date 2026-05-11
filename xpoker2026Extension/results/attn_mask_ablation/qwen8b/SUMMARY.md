# Attention-mask ablation results

- Model: `Qwen/Qwen3-8B`
- Target bucket: `clean_check_or_call`
- n_target processed: 50

Mask: set `attention_mask=0` at the token range covering the `Legal actions: [...]` line of the prompt. This zeros attention TO those key positions across ALL heads at ALL layers.

- **Verb predicted, baseline**: 100.0%
- **Verb predicted, masked**:   100.0%
- **Top-1 ID changed**: 0.0%
- **Top-1 family changed**: 0.0%
- **Mean Δ(CHECK − FOLD)** under mask: -1.711

## Reading guide
- **Verb-predicted-baseline ≫ Verb-predicted-masked** (e.g. 100% → <50%): the model's verb prediction is causally dependent on attending to the legal-actions list. Combined with the Phase J/K finding that dominant heads at L* attend to those positions, the inference is: those heads' attention to the legal-actions tokens is load-bearing for the verb prediction.
- **Top-1 family changed ≥ 50%**: masking reliably shifts the model's verb-distribution. Strong necessity finding.
- **No degradation**: model recovers from the mask via other context. Either the heads have alternate inputs (memorization, structural cues) or the verb prediction doesn't actually need that line.
