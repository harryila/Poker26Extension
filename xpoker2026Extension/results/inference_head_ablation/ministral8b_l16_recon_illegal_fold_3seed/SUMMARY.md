# Inference-time head ablation (behavioral)

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Pipeline: **`recon`** (PromptReconstructor + raw `model.generate` — matches `continuation_after_patch.regenerate_ablated`)
- **Ablation scope:** zeros triplet/control heads at the **last sequence position on every forward pass during action `generate()`** (full CoT reasoning + JSON). This is **more aggressive** than single-position verb patching at L*.
- Filter (recorded_bucket): `illegal_fold`
- n_decisions: 150 (seed=42)
- Triplet heads: `[22, 9, 15]`
- Control heads: `[0, 1, 2]`
- Extended head set: `[9, 15, 22, 24, 30, 31]` (head story: `long_tail_sextet`)

## Aggregate replay rates

| Condition | parseable JSON | illegal_FOLD | clean_CHECK | verb=FOLD | verb=CHECK | verb=BET | verb=UNK |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 100.0% | 82.7% | 9.3% | 82.7% | 9.3% | 8.0% | 0.0% |
| **triplet** | 100.0% | 84.0% | 14.0% | 84.0% | 14.0% | 2.0% | 0.0% |
| **extended** | 100.0% | 80.0% | 19.3% | 80.0% | 19.3% | 0.7% | 0.0% |
| **control** | 100.0% | 58.7% | 39.3% | 58.7% | 39.3% | 2.0% | 0.0% |

## Flip rate on recorded FOLD pool

(records where the **recorded** raw_response parsed as FOLD — either ``illegal_fold`` or ``clean_legal_fold``)

| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |
|---|---:|---:|---:|---:|---:|
| **baseline** | 150 | 9.3% | 8.0% | 17.3% | 0.0% |
| **triplet** | 150 | 14.0% | 2.0% | 16.0% | 0.0% |
| **extended** | 150 | 19.3% | 0.7% | 20.0% | 0.0% |
| **control** | 150 | 39.3% | 2.0% | 41.3% | 0.0% |

**Δ illegal_FOLD (triplet − baseline): +1.3 pp** (see flip-rate table for the cleaner direct measure)
**Net ablation-induced FOLD-flip: triplet 16.0% − baseline 17.3% = -1.3 pp**
- Heads are **behaviorally redundant** for FOLD generation (no large flip beyond baseline). Consistent with §16/Phase O redundancy framing.

## Reading guide
- `parse_fail_rate` should track between conditions; if it spikes under ablation while flip rate stays low, the heads matter for JSON formatting broadly (general damage), not FOLD-specific.
- Compare to `regenerate_ablated` block in `results/continuation_after_patch/{model}/SUMMARY.md` — should agree when `--pipeline recon --filter-recorded-bucket illegal_fold` is set.
