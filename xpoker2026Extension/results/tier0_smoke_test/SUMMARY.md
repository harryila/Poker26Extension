# Tier 0 — Replication smoke test

- Anchor log: `../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl`
- PCE OVERALL records: 371
- UC update records:   58

## Headline number comparison

| Metric | Published anchor | Observed | Pass criterion |
|---|---:|---:|---|
| `|js_cardonly − js_strategyaware|` (OVERALL) | 0.014 ± 0.003 | 0.0163 (raw -0.0163) | ✅ PASS (within ±0.006) |
| mean js_cardonly        | — | 0.4120 | (context) |
| mean js_strategyaware   | — | 0.4283 | (context) |
| mean update correlation r | 0.06 | 0.080 | ✅ PASS (target ±0.10) |

## Reading guide

- **`|js_difference|` PASSES**: extension's PCE pipeline reproduces the published 70B headline (the LLM is ~0.014 nats closer to CardOnly than StrategyAware in JS-divergence). Every downstream finding (8B mech-interp, circuit work, etc.) rests on a sound foundation.
- **`|js_difference|` FAILS**: extension's PCE pipeline diverges from the paper's. Investigate `analysis/compute_pce_distribution.py` and `analysis/oracle_posterior.py` for regressions before trusting any extension-side calibration numbers.
- **mean js_strategyaware ≈ 0.43** is the model's distance from the StrategyAware oracle in absolute terms; this is the headline finding, not the smoke-test target. Smoke-test target is the small but non-zero GAP between CardOnly and StrategyAware, which establishes that the model is closer to CardOnly.
