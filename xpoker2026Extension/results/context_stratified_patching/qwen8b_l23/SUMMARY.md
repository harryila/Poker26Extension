# Context-stratified patching

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Source: `clean_check_or_call` stratified by **`street`**
- Target: `illegal_fold` (n=20)
- Strata run: **4** (skipped strata with <2 sources)

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| FLOP | 176 | 5 | +29.29 | +18.01 | 100.0% |
| PREFLOP | 207 | 5 | +32.88 | +36.78 | 100.0% |
| RIVER | 82 | 5 | +27.59 | +12.04 | 100.0% |
| TURN | 95 | 5 | +28.16 | +21.26 | 100.0% |

**Cross-stratum spec-adj spread: 24.74 nats** (4 strata)
- Patch effect **varies by stratum** → L* mediation is context-modulated.
