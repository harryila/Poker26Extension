# Causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched log: `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layers: [22, 24, 26, 28, 30]

## Controls
- `baseline_top1_match_rate` = 1.0
- `self_patch_max_logit_drift` = 0.0
- `random_source_mean_delta` = -0.702655143447339
- `random_source_n` = 5
- `random_source_test_layer` = 26

## Per-layer effect
| Layer | n | mean Δlogit(CHECK − FOLD) | top-1 flipped to CHECK-family |
|---:|---:|---:|---:|
| 22 | 300 | 10.318 | 100.0% |
| 24 | 300 | 10.606 | 100.0% |
| 26 | 300 | 11.323 | 100.0% |
| 28 | 300 | 11.345 | 100.0% |
| 30 | 300 | 11.426 | 100.0% |
