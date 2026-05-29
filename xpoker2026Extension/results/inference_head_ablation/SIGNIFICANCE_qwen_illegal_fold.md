# Necessity ablation — paired significance (McNemar exact)

- Cells: `results/inference_head_ablation/qwen8b_l*_recon_illegal_fold_*`
- n_cells: 7
- **Control for localization test: L8 whole_attn**
- `flip` = regenerated verb is no longer FOLD (recorded bucket is a FOLD bucket). Tests are paired over shared `(seed, decision_idx)` records.
- `vs baseline`: does ablation flip beyond no-ablation regen drift? `vs control`: is this condition more disruptive than the control ablation (cross-layer or within-cell head control) — the localization test, direction-aware.

| Layer | scope | n | base flip | abl flip | net pp | McNemar vs baseline | McNemar vs control | parse_fail |
|---|---|---:|---:|---:|---:|---|---|---:|
| L8 | whole-attn | 24 | 33.3% | 75.0% | +41.7 | 11/1 disc, p=6.35e-03 | — (is control) | 0.0% |
| L18 | whole-attn | 24 | 33.3% | 75.0% | +41.7 | 11/1 disc, p=6.35e-03 | 4/4 disc, p=1.00e+00 (NS) [xlayer:L8 whole_attn] | 0.0% |
| L19 | whole-attn | 24 | 33.3% | 91.7% | +58.3 | 14/0 disc, p=1.22e-04 | 5/1 disc, p=2.19e-01 (NS) [xlayer:L8 whole_attn] | 0.0% |
| L19 | topk_l19 | 24 | 33.3% | 91.7% | +58.3 | 15/1 disc, p=5.19e-04 | 6/2 disc, p=2.89e-01 (NS) [xlayer:L8 whole_attn] | 0.0% |
| L20 | whole-attn | 24 | 33.3% | 70.8% | +37.5 | 14/5 disc, p=6.36e-02 | 4/5 disc, p=1.00e+00 (NS) [xlayer:L8 whole_attn] | 0.0% |
| L20 | topk_l20 | 24 | 33.3% | 62.5% | +29.2 | 10/3 disc, p=9.23e-02 | 2/5 disc, p=4.53e-01 (NS) [xlayer:L8 whole_attn] | 0.0% |
| L23 | whole-attn | 24 | 33.3% | 79.2% | +45.8 | 12/1 disc, p=3.42e-03 | 3/2 disc, p=1.00e+00 (NS) [xlayer:L8 whole_attn] | 0.0% |

## Reading
- A genuine, *localized* necessity result = `✓ sig necessary` (p<0.05 AND the condition flips MORE than control) with low `parse_fail`. The condition's heads/attention are necessary for FOLD over-and-above the generic disruption captured by control.
- `✗ REVERSED` = p<0.05 but the CONTROL flips more than the condition — the *opposite* of necessity (the named heads matter less than an arbitrary control; e.g. the Ministral §9 null).
- `McNemar vs baseline` significant but `vs control` NS = generic disruption, not layer/head-specific necessity.
- discordant counts `a/b`: `a` = records flipped by this condition but not the comparison; `b` = the reverse. The exact two-sided binomial p-value is computed on `a+b`. The comparison kind is in brackets: `xlayer:` = cross-layer control ablation; `heads:` = within-cell head control.
