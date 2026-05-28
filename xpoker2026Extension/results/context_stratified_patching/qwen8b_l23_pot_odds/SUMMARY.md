# Context-stratified patching

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Source: `clean_check_or_call` stratified by **`pot_odds_quartile`**
- Target: `illegal_fold` (n=20)
- Strata run: **3** (skipped strata with <2 sources)

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| Q1 | 44 | 5 | +25.68 | +14.41 | 100.0% |
| Q2 | 85 | 5 | +25.20 | +29.00 | 100.0% |
| Q4 | 137 | 5 | +32.65 | +18.32 | 100.0% |

**Cross-stratum spec-adj spread: 14.59 nats** (3 strata)
- Patch effect **varies by stratum** → L* mediation is context-modulated.
