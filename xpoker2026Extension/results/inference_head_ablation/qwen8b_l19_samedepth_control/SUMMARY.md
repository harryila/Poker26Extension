# Inference-time head ablation (behavioral)

- Model: `Qwen/Qwen3-8B`
- Layer: **19**
- Pipeline: **`recon`** (PromptReconstructor + raw `model.generate` — matches `continuation_after_patch.regenerate_ablated`)
- **Ablation scope:** zeros triplet/control heads at the **last sequence position on every forward pass during action `generate()`** (full CoT reasoning + JSON). This is **more aggressive** than single-position verb patching at L*.
- Filter (recorded_bucket): `clean_legal_fold`
- n_decisions: 150 (seed=42)
- Triplet heads: `[26, 28, 30]`
- Control heads: `[0, 1, 2]`
- Extended head set: `[26, 28, 30]` (head story: `no_sparse_residual_arrival`)

## Aggregate replay rates

| Condition | parseable JSON | illegal_FOLD | clean_CHECK | verb=FOLD | verb=CHECK | verb=BET | verb=UNK |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 100.0% | 0.0% | 10.0% | 84.7% | 10.0% | 5.3% | 0.0% |
| **top5** | 100.0% | 0.0% | 32.7% | 47.3% | 32.7% | 20.0% | 0.0% |
| **rand5a** | 100.0% | 0.0% | 22.0% | 56.7% | 22.0% | 21.3% | 0.0% |
| **rand5b** | 100.0% | 0.0% | 26.0% | 62.7% | 26.0% | 11.3% | 0.0% |
| **rand5c** | 100.0% | 0.0% | 30.0% | 65.3% | 30.0% | 4.7% | 0.0% |

## Flip rate on recorded FOLD pool

(records where the **recorded** raw_response parsed as FOLD — either ``illegal_fold`` or ``clean_legal_fold``)

| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |
|---|---:|---:|---:|---:|---:|
| **baseline** | 150 | 10.0% | 5.3% | 15.3% | 0.0% |
| **top5** | 150 | 32.7% | 20.0% | 52.7% | 0.0% |
| **rand5a** | 150 | 22.0% | 21.3% | 43.3% | 0.0% |
| **rand5b** | 150 | 26.0% | 11.3% | 37.3% | 0.0% |
| **rand5c** | 150 | 30.0% | 4.7% | 34.7% | 0.0% |

**Net ablation-induced FOLD-flip (top5, heads [31, 3, 21, 1, 0]): 52.7% − baseline 15.3% = +37.3 pp** (parse_fail 0.0%)
- ⚠️ This is flip-over-BASELINE only. A whole-attention/large head ablation is blunt: removing attention at *any* layer biases toward CHECK, so a large net flip can be **generic disruption**, not layer-specific necessity. The necessity claim requires a paired comparison to a CONTROL-layer (or control-head) ablation — see `experiments.necessity_significance` and `results/inference_head_ablation/SIGNIFICANCE_*.md`. Do NOT read this single number as 'necessary'.

**Net ablation-induced FOLD-flip (rand5a, heads [3, 12, 13, 22, 29]): 43.3% − baseline 15.3% = +28.0 pp** (parse_fail 0.0%)
- ⚠️ This is flip-over-BASELINE only. A whole-attention/large head ablation is blunt: removing attention at *any* layer biases toward CHECK, so a large net flip can be **generic disruption**, not layer-specific necessity. The necessity claim requires a paired comparison to a CONTROL-layer (or control-head) ablation — see `experiments.necessity_significance` and `results/inference_head_ablation/SIGNIFICANCE_*.md`. Do NOT read this single number as 'necessary'.

**Net ablation-induced FOLD-flip (rand5b, heads [12, 24, 25, 26, 30]): 37.3% − baseline 15.3% = +22.0 pp** (parse_fail 0.0%)
- ⚠️ This is flip-over-BASELINE only. A whole-attention/large head ablation is blunt: removing attention at *any* layer biases toward CHECK, so a large net flip can be **generic disruption**, not layer-specific necessity. The necessity claim requires a paired comparison to a CONTROL-layer (or control-head) ablation — see `experiments.necessity_significance` and `results/inference_head_ablation/SIGNIFICANCE_*.md`. Do NOT read this single number as 'necessary'.

**Net ablation-induced FOLD-flip (rand5c, heads [2, 5, 12, 15, 16]): 34.7% − baseline 15.3% = +19.3 pp** (parse_fail 0.0%)
- ⚠️ This is flip-over-BASELINE only. A whole-attention/large head ablation is blunt: removing attention at *any* layer biases toward CHECK, so a large net flip can be **generic disruption**, not layer-specific necessity. The necessity claim requires a paired comparison to a CONTROL-layer (or control-head) ablation — see `experiments.necessity_significance` and `results/inference_head_ablation/SIGNIFICANCE_*.md`. Do NOT read this single number as 'necessary'.

## Reading guide
- `parse_fail_rate` should track between conditions; if it spikes under ablation while flip rate stays low, the heads matter for JSON formatting broadly (general damage), not FOLD-specific.
- Compare to `regenerate_ablated` block in `results/continuation_after_patch/{model}/SUMMARY.md` — should agree when `--pipeline recon --filter-recorded-bucket illegal_fold` is set.
