# Context-stratified patching

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Source: `clean_check_or_call` stratified by **`street`**
- Target: `illegal_fold` (n=20)
- Strata run: **2** (skipped strata with <2 sources)

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| FLOP | 20 | 5 | +7.61 | +5.10 | 100.0% |
| PREFLOP | 12 | 5 | +9.43 | +10.91 | 100.0% |

**Cross-stratum spec-adj spread: 5.80 nats** (2 strata)
- Patch effect **varies by stratum** → L* mediation is context-modulated.
