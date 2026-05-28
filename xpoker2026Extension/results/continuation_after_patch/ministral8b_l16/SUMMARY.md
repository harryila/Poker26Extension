# Continuation after L* patch

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Source bucket: `clean_check_or_call` → target: `illegal_fold`
- n_targets: 25 | n_sources: 5 | patch pairs: 125
- Ablated heads (regenerate_ablated): `[22, 9, 15]`
- Continue tokens (greedy after verb): 180

## Response quality

| Mode | denominator | coherent CoT+JSON | valid JSON only | broken | empty |
|---|---:|---:|---:|---:|---:|
| Recorded log | 25 | 100% | 0% | 0% | 0% |
| Full regenerate | 25 | 100% | 0% | 0% | 0% |
| Full regenerate + ablation | 25 | 100% | 0% | 0% | 0% |
| Patch verb + continue (× sources) | 125 | 100% | 0% | 0% | 0% |

## Verb distribution (parsed `"action": "..."`)

| Mode | denominator | FOLD | CHECK_OR_CALL | BET_OR_RAISE | UNK (parse fail) |
|---|---:|---:|---:|---:|---:|
| Recorded log | 25 | 100% | 0% | 0% | 0% |
| Full regenerate | 25 | 96% | 0% | 4% | 0% |
| Full regenerate + ablation | 25 | 84% | 12% | 4% | 0% |
| Patch verb + continue (× sources) | 125 | 0% | 100% | 0% | 0% |

## Flip rate on recorded-FOLD targets (n=25)

| Mode | FOLD→CHECK | parse fail (UNK) |
|---|---:|---:|
| **regenerate_baseline** | 0.0% (0) | 0.0% (0) |
| **regenerate_ablated**  | 12.0% (3) | 0.0% (0) |

**Net ablation flip (FOLD→CHECK over baseline): +12.0 pp**  | **Net parse-fail damage: +0.0 pp**

**Patch flip (single-forward, source×target pairs): 125/125 = 100.0% to CHECK_OR_CALL**

## Interpretation
- Compare **`regenerate_ablated`** flip rate to `results/inference_head_ablation/<model>_l*_recon_illegal_fold/SUMMARY.md` — numbers should agree (same prompt path, same hook scope, same filter).
- If **net parse-fail damage** is large, ablation hurts JSON generation broadly (Llama L=14 case) — net flip should be interpreted as 'general damage' not 'surgical FOLD necessity'.
- **patch_verb_then_continue** aggregates over every (source × target) pair — surgical 1-token verb swap, doesn't disrupt JSON close.

See `examples.jsonl` (one row per source×target for patch mode).
