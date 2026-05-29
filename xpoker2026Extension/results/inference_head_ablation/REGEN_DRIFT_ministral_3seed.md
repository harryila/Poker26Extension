# Baseline regeneration drift per ablation cell (necessity reliability)

`drift` = fraction of recorded-FOLD targets whose **no-ablation** regeneration is already non-FOLD (pure T=0 nondeterminism). Necessity (ablation−baseline) is only trustworthy where drift is LOW and parse_fail ≈ 0.

| cell | n | baseline drift | parse_fail | necessity reliability |
|---|---:|---:|---:|---|
| ministral8b_l16_recon_illegal_fold_3seed | 150 | 17.3% | 0.0% | OK (low drift) |

## Reading
- **Low drift (Qwen clean_legal_fold ~15%)** → necessity delta is real; this is why the headline Qwen necessity uses the clean_legal_fold pool.
- **High drift (Llama illegal_fold ~73%)** → the verb is already unstable on plain regeneration; necessity must be read as a *control-paired* McNemar delta, not a raw ablated flip rate, and continuation-based necessity is unmeasurable for Llama.
- parse_fail ≈ 0 everywhere confirms flips are coherent CoT+JSON decisions, not broken generations — the earlier HFAgent 5%-parse confound does not recur under recon.
