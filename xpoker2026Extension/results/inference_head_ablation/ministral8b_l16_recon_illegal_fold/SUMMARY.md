# Inference-time head ablation (behavioral)

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Pipeline: **`recon`** (PromptReconstructor + raw `model.generate` — matches `continuation_after_patch.regenerate_ablated`)
- **Ablation scope:** zeros triplet/control heads at the **last sequence position on every forward pass during action `generate()`** (full CoT reasoning + JSON). This is **more aggressive** than single-position verb patching at L*.
- Filter (recorded_bucket): `illegal_fold`
- n_decisions: 80 (seed=42)
- Triplet heads: `[22, 9, 15]`
- Control heads: `[0, 1, 2]`
- Extended head set: `[9, 15, 22, 24, 30, 31]` (head story: `long_tail_sextet`)

## Aggregate replay rates

| Condition | parseable JSON | illegal_FOLD | clean_CHECK | verb=FOLD | verb=CHECK | verb=BET | verb=UNK |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 100.0% | 78.8% | 11.2% | 78.8% | 11.2% | 10.0% | 0.0% |
| **triplet** | 100.0% | 81.2% | 17.5% | 81.2% | 17.5% | 1.2% | 0.0% |
| **control** | 100.0% | 56.2% | 42.5% | 56.2% | 42.5% | 1.2% | 0.0% |

## Flip rate on recorded FOLD pool

(records where the **recorded** raw_response parsed as FOLD — either ``illegal_fold`` or ``clean_legal_fold``)

| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |
|---|---:|---:|---:|---:|---:|
| **baseline** | 80 | 11.2% | 10.0% | 21.2% | 0.0% |
| **triplet** | 80 | 17.5% | 1.2% | 18.8% | 0.0% |
| **control** | 80 | 42.5% | 1.2% | 43.8% | 0.0% |

**Δ illegal_FOLD (triplet − baseline): +2.5 pp** (see flip-rate table for the cleaner direct measure)
**Net ablation-induced FOLD-flip: triplet 18.8% − baseline 21.2% = -2.5 pp**
- Heads are **behaviorally redundant** for FOLD generation (no large flip beyond baseline). Consistent with §16/Phase O redundancy framing.

## Reading guide
- `parse_fail_rate` should track between conditions; if it spikes under ablation while flip rate stays low, the heads matter for JSON formatting broadly (general damage), not FOLD-specific.
- Compare to `regenerate_ablated` block in `results/continuation_after_patch/{model}/SUMMARY.md` — should agree when `--pipeline recon --filter-recorded-bucket illegal_fold` is set.
