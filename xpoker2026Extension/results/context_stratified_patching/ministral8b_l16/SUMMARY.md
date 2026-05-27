# Context-stratified patching

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Source: `clean_check_or_call` stratified by pot-odds quartile
- Target: `illegal_fold` (n=20)

| Stratum | n_src | mean Δ | spec-adj Δ | top-1 flip | pot odds range |
|---|---:|---:|---:|---:|---|
| Q4 | 5 | +8.32 | +5.90 | 100.0% | [0.00, 0.00] |

**Cross-stratum spec-adj spread: 0.00 nats**
- Patch effect is **stable across pot-odds contexts** → circuit behaves like a verb encoder downstream of equity stratification.
