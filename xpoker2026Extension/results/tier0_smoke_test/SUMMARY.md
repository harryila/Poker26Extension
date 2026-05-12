# Tier 0 — Replication smoke test

- Anchor log: `../logs/phase2_70b_t0_s42_informative_v2_enriched.jsonl`
- PCE records read: 0

## Headline number comparison

| Metric | Published anchor | Observed | Pass criterion |
|---|---:|---:|---|
| mean js_to_strategy_aware | 0.014 ± 0.003 | (could not extract) | ❓ inconclusive |
| update correlation r | 0.06 | (could not extract) | ❓ inconclusive |

## Reading guide

- **All ✅ PASS**: extension's analysis pipeline reproduces published 70B numbers within tolerance. Every downstream finding (8B mech-interp, circuit work, etc.) rests on a sound foundation.
- **❌ FAIL on mean_js**: extension's PCE pipeline diverges from the paper's. Investigate  for regressions before trusting any extension-side calibration numbers.
- **❓ inconclusive**: column names in the CSV don't match expected. Check the CSV manually and update this script's column-name search.
