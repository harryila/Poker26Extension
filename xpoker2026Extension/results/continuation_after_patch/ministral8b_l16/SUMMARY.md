# Continuation after L* patch

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Source bucket: `clean_check_or_call` → target: `illegal_fold`
- n_targets: 25 | n_sources: 5 | patch pairs: 125
- Ablated heads (regenerate_ablated): `[22, 9, 15]`
- Continue tokens (greedy after verb): 80

## Response quality

| Mode | denominator | coherent CoT+JSON | valid JSON only | broken | empty |
|---|---:|---:|---:|---:|---:|
| Recorded log | 25 | 100% | 0% | 0% | 0% |
| Full regenerate | 25 | 88% | 0% | 12% | 0% |
| Full regenerate + ablation | 25 | 72% | 0% | 28% | 0% |
| Patch verb + continue (× sources) | 125 | 100% | 0% | 0% | 0% |

## Interpretation
- **patch_verb_then_continue** aggregates over every (source × target) pair — not a single source residual.
- If patch-continue is mostly `broken_json` but regenerate_ablated is `coherent_cot_json`, the one-forward patch does not sustain global coherence (circuit acts during full decoding).

See `examples.jsonl` (one row per source×target for patch mode).
