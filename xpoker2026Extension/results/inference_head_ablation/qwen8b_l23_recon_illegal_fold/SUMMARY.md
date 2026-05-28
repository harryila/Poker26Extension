# Inference-time head ablation (behavioral)

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Pipeline: **`recon`** (PromptReconstructor + raw `model.generate` — matches `continuation_after_patch.regenerate_ablated`)
- **Ablation scope:** zeros triplet/control heads at the **last sequence position on every forward pass during action `generate()`** (full CoT reasoning + JSON). This is **more aggressive** than single-position verb patching at L*.
- Filter (recorded_bucket): `illegal_fold`
- n_decisions: 24 (seed=42)
- Triplet heads: `[26, 28, 30]`
- Control heads: `[0, 1, 2]`
- Extended head set: `[26, 28, 30]` (head story: `no_sparse_residual_arrival`)

## Aggregate replay rates

| Condition | parseable JSON | illegal_FOLD | clean_CHECK | verb=FOLD | verb=CHECK | verb=BET | verb=UNK |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 100.0% | 66.7% | 25.0% | 66.7% | 25.0% | 8.3% | 0.0% |
| **triplet** | 100.0% | 8.3% | 83.3% | 8.3% | 83.3% | 8.3% | 0.0% |
| **control** | 100.0% | 41.7% | 41.7% | 41.7% | 41.7% | 16.7% | 0.0% |

## Flip rate on recorded FOLD pool

(records where the **recorded** raw_response parsed as FOLD — either ``illegal_fold`` or ``clean_legal_fold``)

| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |
|---|---:|---:|---:|---:|---:|
| **baseline** | 24 | 25.0% | 8.3% | 33.3% | 0.0% |
| **triplet** | 24 | 83.3% | 8.3% | 91.7% | 0.0% |
| **control** | 24 | 41.7% | 16.7% | 58.3% | 0.0% |

**Δ illegal_FOLD (triplet − baseline): -58.3 pp** (see flip-rate table for the cleaner direct measure)
**Net ablation-induced FOLD-flip: triplet 91.7% − baseline 33.3% = +58.3 pp**
- Heads are **behaviorally necessary** for the FOLD action: ablating them flips a large fraction of recorded-FOLD decisions to CHECK/BET.

## Reading guide
- `parse_fail_rate` should track between conditions; if it spikes under ablation while flip rate stays low, the heads matter for JSON formatting broadly (general damage), not FOLD-specific.
- Compare to `regenerate_ablated` block in `results/continuation_after_patch/{model}/SUMMARY.md` — should agree when `--pipeline recon --filter-recorded-bucket illegal_fold` is set.
