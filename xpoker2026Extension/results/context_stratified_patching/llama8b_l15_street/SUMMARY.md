# Context-stratified patching

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **15**
- Source: `clean_check_or_call` stratified by **`street`**
- Target: `illegal_fold` (n=20)
- Strata run: **4** (skipped strata with <2 sources)

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| FLOP | 247 | 5 | +12.43 | +4.78 | 95.0% |
| PREFLOP | 190 | 5 | +13.45 | +9.13 | 95.0% |
| RIVER | 34 | 5 | +12.28 | +13.05 | 95.0% |
| TURN | 86 | 5 | +12.32 | +11.11 | 95.0% |

**Cross-stratum spec-adj spread: 8.27 nats** (4 strata)
- Patch effect **varies by stratum** → L* mediation is context-modulated.
