# Context-stratified patching

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Source: `clean_check_or_call` stratified by **`pot_odds_quartile`**
- Target: `illegal_fold` (n=20)
- Strata run: **3** (skipped strata with <2 sources)

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| Q1 | 3 | 3 | +7.21 | +1.55 | 80.0% |
| Q2 | 98 | 5 | +7.61 | +4.72 | 85.0% |
| Q4 | 153 | 5 | +12.08 | +12.60 | 95.0% |

**Cross-stratum spec-adj spread: 11.05 nats** (3 strata)
- Patch effect **varies by stratum** → L* mediation is context-modulated.
