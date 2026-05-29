# Necessity ablation — paired significance (McNemar exact)

- Cells: `results/inference_head_ablation/llama8b_l14_recon_illegal_fold`
- n_cells: 2
- `flip` = regenerated verb is no longer FOLD (recorded bucket is a FOLD bucket). Tests are paired over shared `(seed, decision_idx)` records.
- `vs baseline`: does ablation flip beyond no-ablation regen drift? `vs control`: is this condition more disruptive than the control ablation (cross-layer or within-cell head control) — the localization test, direction-aware.

| Layer | scope | n | base flip | abl flip | net pp | McNemar vs baseline | McNemar vs control | parse_fail |
|---|---|---:|---:|---:|---:|---|---|---:|
| L14 | triplet | 68 | 73.5% | 97.1% | +23.5 | 16/0 disc, p=3.05e-05 | 20/1 disc, p=2.10e-05 (✓ sig necessary) [heads:control] | 0.0% |
| L14 | control | 68 | 73.5% | 69.1% | -4.4 | 13/16 disc, p=7.11e-01 | — (is control) | 0.0% |

## Reading
- A genuine, *localized* necessity result = `✓ sig necessary` (p<0.05 AND the condition flips MORE than control) with low `parse_fail`. The condition's heads/attention are necessary for FOLD over-and-above the generic disruption captured by control.
- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the condition — the *opposite* of necessity (the named heads matter less than an arbitrary control; e.g. the Ministral §9 null).
- `McNemar vs baseline` significant but `vs control` NS = generic disruption, not layer/head-specific necessity.
- discordant counts `a/b`: `a` = records flipped by this condition but not the comparison; `b` = the reverse. The exact two-sided binomial p-value is computed on `a+b`. The comparison kind is in brackets: `xlayer:` = cross-layer control ablation; `heads:` = within-cell head control.
