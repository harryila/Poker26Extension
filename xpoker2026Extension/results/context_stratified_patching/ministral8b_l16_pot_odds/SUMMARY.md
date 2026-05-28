# Context-stratified patching

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Source: `clean_check_or_call` stratified by **`pot_odds_quartile`**
- Target: `illegal_fold` (n=20)
- Strata run: **1** (skipped strata with <2 sources)

- ⚠️ fewer than 8 clean_check_or_call sources with bet_to_call>0; pot_odds_quartile strata may be sparse

| Stratum | pool n | n_src | mean Δ | spec-adj Δ | top-1 flip |
|---|---:|---:|---:|---:|---:|
| Q4 | 6 | 5 | +8.17 | +7.65 | 100.0% |

- ⚠️ Only one stratum met `min_sources_per_stratum`; cross-stratum comparison not valid. Try `--stratify-by street`.
