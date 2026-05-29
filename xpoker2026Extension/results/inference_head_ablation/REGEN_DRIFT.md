# Baseline regeneration drift per ablation cell (necessity reliability)

`drift` = fraction of recorded-FOLD targets whose **no-ablation** regeneration is already non-FOLD (pure T=0 nondeterminism). Necessity (ablation−baseline) is only trustworthy where drift is LOW and parse_fail ≈ 0.

| cell | n | baseline drift | parse_fail | necessity reliability |
|---|---:|---:|---:|---|
| llama8b_l14_recon_illegal_fold | 68 | 73.5% | 0.0% | UNRELIABLE (drift swamps effect) |
| llama8b_l15_recon_illegal_fold_negctrl | 68 | 73.5% | 0.0% | UNRELIABLE (drift swamps effect) |
| ministral8b_l16_cot | 200 | 100.0% | 0.0% | UNRELIABLE (drift swamps effect) |
| ministral8b_l16_recon_illegal_fold | 80 | 21.2% | 0.0% | OK (low drift) |
| ministral8b_l16_recon_illegal_fold_sextet | 80 | 21.2% | 0.0% | OK (low drift) |
| qwen8b_l18_recon_clean_legal_fold_wholeattn | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l18_recon_illegal_fold_wholeattn | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l19_recon_clean_legal_fold_topk | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l19_recon_clean_legal_fold_wholeattn | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l19_recon_illegal_fold_topk | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l19_recon_illegal_fold_wholeattn | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l20_recon_clean_legal_fold_topk | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l20_recon_clean_legal_fold_wholeattn | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l20_recon_illegal_fold_topk | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l20_recon_illegal_fold_wholeattn | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l23_recon_clean_legal_fold_wholeattn | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l23_recon_illegal_fold | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l23_recon_illegal_fold_wholeattn | 24 | 33.3% | 0.0% | marginal |
| qwen8b_l8_recon_clean_legal_fold_wholeattn | 150 | 15.3% | 0.0% | OK (low drift) |
| qwen8b_l8_recon_illegal_fold_wholeattn | 24 | 33.3% | 0.0% | marginal |

## Reading
- **Low drift (Qwen clean_legal_fold ~15%)** → necessity delta is real; this is why the headline Qwen necessity uses the clean_legal_fold pool.
- **High drift (Llama illegal_fold ~73%)** → the verb is already unstable on plain regeneration; necessity must be read as a *control-paired* McNemar delta, not a raw ablated flip rate, and continuation-based necessity is unmeasurable for Llama.
- parse_fail ≈ 0 everywhere confirms flips are coherent CoT+JSON decisions, not broken generations — the earlier HFAgent 5%-parse confound does not recur under recon.
