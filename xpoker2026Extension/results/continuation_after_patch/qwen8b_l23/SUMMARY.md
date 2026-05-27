# Continuation after L* patch

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Source bucket: `clean_check_or_call` → target: `illegal_fold`
- n_targets: 24 | n_sources: 5 | patch pairs: 120
- Ablated heads (regenerate_ablated): `[26, 28, 30]`
- Continue tokens (greedy after verb): 180

## Response quality

| Mode | denominator | coherent CoT+JSON | valid JSON only | broken | empty |
|---|---:|---:|---:|---:|---:|
| Recorded log | 24 | 100% | 0% | 0% | 0% |
| Full regenerate | 24 | 100% | 0% | 0% | 0% |
| Full regenerate + ablation | 24 | 100% | 0% | 0% | 0% |
| Patch verb + continue (× sources) | 120 | 100% | 0% | 0% | 0% |

## Interpretation
- **patch_verb_then_continue** aggregates over every (source × target) pair — not a single source residual.
- If patch-continue is mostly `broken_json` but regenerate_ablated is `coherent_cot_json`, the one-forward patch does not sustain global coherence (circuit acts during full decoding).

See `examples.jsonl` (one row per source×target for patch mode).
