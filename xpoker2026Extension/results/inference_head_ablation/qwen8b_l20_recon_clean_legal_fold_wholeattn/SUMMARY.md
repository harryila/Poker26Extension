# Inference-time head ablation (behavioral)

- Model: `Qwen/Qwen3-8B`
- Layer: **20**
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
| **whole_attn** | 100.0% | 0.0% | 13.3% | 73.3% | 13.3% | 13.3% | 0.0% |

## Flip rate on recorded FOLD pool

(records where the **recorded** raw_response parsed as FOLD — either ``illegal_fold`` or ``clean_legal_fold``)

| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |
|---|---:|---:|---:|---:|---:|
| **baseline** | 150 | 10.0% | 5.3% | 15.3% | 0.0% |
| **whole_attn** | 150 | 13.3% | 13.3% | 26.7% | 0.0% |

**Net ablation-induced FOLD-flip (whole_attn, whole-attention block): 26.7% − baseline 15.3% = +11.3 pp** (parse_fail 0.0%)
- Moderate necessity for `whole_attn`; check parse_fail.

## Reading guide
- `parse_fail_rate` should track between conditions; if it spikes under ablation while flip rate stays low, the heads matter for JSON formatting broadly (general damage), not FOLD-specific.
- Compare to `regenerate_ablated` block in `results/continuation_after_patch/{model}/SUMMARY.md` — should agree when `--pipeline recon --filter-recorded-bucket illegal_fold` is set.
