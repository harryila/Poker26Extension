# Necessity ablation — paired significance (McNemar exact)

- Cells: `results/inference_head_ablation/qwen8b_l19_samedepth_control`
- n_cells: 4
- `flip` = regenerated verb is no longer FOLD (recorded bucket is a FOLD bucket). Tests are paired over shared `(seed, decision_idx)` records.
- `vs baseline`: does ablation flip beyond no-ablation regen drift? `vs control`: is this condition more disruptive than the control ablation (cross-layer or within-cell head control) — the localization test, direction-aware.

| Layer | scope | n | base flip | abl flip | net pp | McNemar vs baseline | McNemar vs control | parse_fail |
|---|---|---:|---:|---:|---:|---|---|---:|
| L19 | top5 | 150 | 15.3% | 52.7% | +37.3 | 62/6 disc, p=8.18e-13 | 31/17 disc, p=5.95e-02 (~marg) [heads:rand5a] | 0.0% |
| L19 | rand5a | 150 | 15.3% | 43.3% | +28.0 | 48/6 disc, p=3.26e-09 | — (is control) | 0.0% |
| L19 | rand5b | 150 | 15.3% | 37.3% | +22.0 | 44/11 disc, p=8.70e-06 | 22/31 disc, p=2.72e-01 (NS) [heads:rand5a] | 0.0% |
| L19 | rand5c | 150 | 15.3% | 34.7% | +19.3 | 36/7 disc, p=8.96e-06 | 14/27 disc, p=5.96e-02 (~marg-reversed) [heads:rand5a] | 0.0% |

## Reading
- A genuine, *localized* necessity result = `✓ sig necessary` (p<0.05 AND the condition flips MORE than control) with low `parse_fail`. The condition's heads/attention are necessary for FOLD over-and-above the generic disruption captured by control.
- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the condition — the *opposite* of necessity (the named heads matter less than an arbitrary control; e.g. the Ministral §9 null).
- `McNemar vs baseline` significant but `vs control` NS = generic disruption, not layer/head-specific necessity.
- discordant counts `a/b`: `a` = records flipped by this condition but not the comparison; `b` = the reverse. The exact two-sided binomial p-value is computed on `a+b`. The comparison kind is in brackets: `xlayer:` = cross-layer control ablation; `heads:` = within-cell head control.
