# Necessity ablation — paired significance (McNemar exact)

- Cells: `results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_sextet`
- n_cells: 3
- `flip` = regenerated verb is no longer FOLD (recorded bucket is a FOLD bucket). Tests are paired over shared `(seed, decision_idx)` records.
- `vs baseline`: does ablation flip beyond no-ablation regen drift? `vs control`: is this condition more disruptive than the control ablation (cross-layer or within-cell head control) — the localization test, direction-aware.

| Layer | scope | n | base flip | abl flip | net pp | McNemar vs baseline | McNemar vs control | parse_fail |
|---|---|---:|---:|---:|---:|---|---|---:|
| L16 | triplet | 80 | 21.2% | 18.8% | -2.5 | 8/10 disc, p=8.15e-01 | 7/27 disc, p=8.21e-04 (✗ REVERSED (control flips MORE)) [heads:control] | 0.0% |
| L16 | control | 80 | 21.2% | 43.8% | +22.5 | 24/6 disc, p=1.43e-03 | — (is control) | 0.0% |
| L16 | extended | 80 | 21.2% | 22.5% | +1.3 | 9/8 disc, p=1.00e+00 | 6/23 disc, p=2.32e-03 (✗ REVERSED (control flips MORE)) [heads:control] | 0.0% |

## Reading
- A genuine, *localized* necessity result = `✓ sig necessary` (p<0.05 AND the condition flips MORE than control) with low `parse_fail`. The condition's heads/attention are necessary for FOLD over-and-above the generic disruption captured by control.
- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the condition — the *opposite* of necessity (the named heads matter less than an arbitrary control; e.g. the Ministral §9 null).
- `McNemar vs baseline` significant but `vs control` NS = generic disruption, not layer/head-specific necessity.
- discordant counts `a/b`: `a` = records flipped by this condition but not the comparison; `b` = the reverse. The exact two-sided binomial p-value is computed on `a+b`. The comparison kind is in brackets: `xlayer:` = cross-layer control ablation; `heads:` = within-cell head control.
