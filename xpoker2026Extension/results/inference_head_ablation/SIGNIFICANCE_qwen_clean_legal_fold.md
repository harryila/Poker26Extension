# Necessity ablation — paired significance (McNemar exact)

- Cells: `results/inference_head_ablation/qwen8b_l*_recon_clean_legal_fold_*`
- n_cells: 7
- **Control for localization test: L8 whole_attn**
- `flip` = regenerated verb is no longer FOLD (recorded bucket is a FOLD bucket). Tests are paired over shared `(seed, decision_idx)` records.
- `vs baseline`: does ablation flip beyond no-ablation regen drift? `vs control`: is this condition more disruptive than the control ablation (cross-layer or within-cell head control) — the localization test, direction-aware.

| Layer | scope | n | base flip | abl flip | net pp | McNemar vs baseline | McNemar vs control | parse_fail |
|---|---|---:|---:|---:|---:|---|---|---:|
| L8 | whole-attn | 150 | 15.3% | 36.0% | +20.7 | 39/8 disc, p=5.54e-06 | — (is control) | 0.0% |
| L18 | whole-attn | 150 | 15.3% | 46.0% | +30.7 | 55/9 disc, p=3.54e-09 | 39/24 disc, p=7.69e-02 (~marg) [xlayer:L8 whole_attn] | 0.0% |
| L19 | whole-attn | 150 | 15.3% | 58.0% | +42.7 | 70/6 disc, p=6.31e-15 | 48/15 disc, p=3.76e-05 (✓ sig necessary) [xlayer:L8 whole_attn] | 0.0% |
| L19 | topk_l19 | 150 | 15.3% | 52.7% | +37.3 | 62/6 disc, p=8.18e-13 | 40/15 disc, p=1.02e-03 (✓ sig necessary) [xlayer:L8 whole_attn] | 0.0% |
| L20 | whole-attn | 150 | 15.3% | 26.7% | +11.3 | 32/15 disc, p=1.86e-02 | 19/33 disc, p=7.04e-02 (~marg-reversed) [xlayer:L8 whole_attn] | 0.0% |
| L20 | topk_l20 | 150 | 15.3% | 40.0% | +24.7 | 46/9 disc, p=4.34e-07 | 30/24 disc, p=4.97e-01 (NS) [xlayer:L8 whole_attn] | 0.0% |
| L23 | whole-attn | 150 | 15.3% | 31.3% | +16.0 | 34/10 disc, p=3.88e-04 | 18/25 disc, p=3.60e-01 (NS) [xlayer:L8 whole_attn] | 0.0% |

## Reading
- A genuine, *localized* necessity result = `✓ sig necessary` (p<0.05 AND the condition flips MORE than control) with low `parse_fail`. The condition's heads/attention are necessary for FOLD over-and-above the generic disruption captured by control.
- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the condition — the *opposite* of necessity (the named heads matter less than an arbitrary control; e.g. the Ministral §9 null).
- `McNemar vs baseline` significant but `vs control` NS = generic disruption, not layer/head-specific necessity.
- discordant counts `a/b`: `a` = records flipped by this condition but not the comparison; `b` = the reverse. The exact two-sided binomial p-value is computed on `a+b`. The comparison kind is in brackets: `xlayer:` = cross-layer control ablation; `heads:` = within-cell head control.
