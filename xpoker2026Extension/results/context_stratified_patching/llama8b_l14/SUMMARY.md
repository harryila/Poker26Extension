# Context-stratified patching

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Source: `clean_check_or_call` stratified by **`street`**
- Target: `illegal_fold` (n=20)
- Strata run: **4** (skipped strata with <2 sources)

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| FLOP | 247 | 5 | +9.48 | +3.83 | 86.0% |
| PREFLOP | 190 | 5 | +10.91 | +8.02 | 95.0% |
| RIVER | 34 | 5 | +8.72 | +9.23 | 91.0% |
| TURN | 86 | 5 | +8.65 | +7.53 | 89.0% |

**Cross-stratum spec-adj spread: 5.41 nats** (4 strata)
- Patch effect **varies by stratum** → L* mediation is context-modulated.
