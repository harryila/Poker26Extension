# Necessity ablation — paired significance (McNemar exact)

- Cells: `results/inference_head_ablation/ministral8b_l16_recon_illegal_fold_3seed`
- n_cells: 3
- `flip` = regenerated verb is no longer FOLD (recorded bucket is a FOLD bucket). Tests are paired over shared `(seed, decision_idx)` records.
- `vs baseline`: does ablation flip beyond no-ablation regen drift? `vs control`: is this condition more disruptive than the control ablation (cross-layer or within-cell head control) — the localization test, direction-aware.

| Layer | scope | n | base flip | abl flip | net pp | McNemar vs baseline | McNemar vs control | parse_fail |
|---|---|---:|---:|---:|---:|---|---|---:|
| L16 | triplet | 150 | 17.3% | 16.0% | -1.3 | 15/17 disc, p=8.60e-01 | 13/51 disc, p=1.88e-06 (✗ REVERSED (control flips MORE)) [heads:control] | 0.0% |
| L16 | control | 150 | 17.3% | 41.3% | +24.0 | 46/10 disc, p=1.25e-06 | — (is control) | 0.0% |
| L16 | extended | 150 | 17.3% | 20.0% | +2.7 | 18/14 disc, p=5.97e-01 | 14/46 disc, p=4.22e-05 (✗ REVERSED (control flips MORE)) [heads:control] | 0.0% |

## Reading
- A genuine, *localized* necessity result = `✓ sig necessary` (p<0.05 AND the condition flips MORE than control) with low `parse_fail`. The condition's heads/attention are necessary for FOLD over-and-above the generic disruption captured by control.
- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the condition — the *opposite* of necessity (the named heads matter less than an arbitrary control; e.g. the Ministral §9 null).
- `McNemar vs baseline` significant but `vs control` NS = generic disruption, not layer/head-specific necessity.
- discordant counts `a/b`: `a` = records flipped by this condition but not the comparison; `b` = the reverse. The exact two-sided binomial p-value is computed on `a+b`. The comparison kind is in brackets: `xlayer:` = cross-layer control ablation; `heads:` = within-cell head control.
