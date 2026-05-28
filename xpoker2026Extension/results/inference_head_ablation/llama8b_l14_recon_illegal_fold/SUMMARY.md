# Inference-time head ablation (behavioral)

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Pipeline: **`recon`** (PromptReconstructor + raw `model.generate` — matches `continuation_after_patch.regenerate_ablated`)
- **Ablation scope:** zeros triplet/control heads at the **last sequence position on every forward pass during action `generate()`** (full CoT reasoning + JSON). This is **more aggressive** than single-position verb patching at L*.
- Filter (recorded_bucket): `illegal_fold`
- n_decisions: 68 (seed=42)
- Triplet heads: `[5, 23, 24]`
- Control heads: `[0, 1, 2]`
- Extended head set: `[5, 23, 24]` (head story: `sparse_triplet`)

## Aggregate replay rates

| Condition | parseable JSON | illegal_FOLD | clean_CHECK | verb=FOLD | verb=CHECK | verb=BET | verb=UNK |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 100.0% | 26.5% | 50.0% | 26.5% | 50.0% | 23.5% | 0.0% |
| **triplet** | 100.0% | 2.9% | 52.9% | 2.9% | 52.9% | 44.1% | 0.0% |
| **control** | 100.0% | 30.9% | 38.2% | 30.9% | 38.2% | 30.9% | 0.0% |

## Flip rate on recorded FOLD pool

(records where the **recorded** raw_response parsed as FOLD — either ``illegal_fold`` or ``clean_legal_fold``)

| Condition | n | FOLD→CHECK | FOLD→BET | any flip | parse fail |
|---|---:|---:|---:|---:|---:|
| **baseline** | 68 | 50.0% | 23.5% | 73.5% | 0.0% |
| **triplet** | 68 | 52.9% | 44.1% | 97.1% | 0.0% |
| **control** | 68 | 38.2% | 30.9% | 69.1% | 0.0% |

**Δ illegal_FOLD (triplet − baseline): -23.5 pp** (see flip-rate table for the cleaner direct measure)
**Net ablation-induced FOLD-flip: triplet 97.1% − baseline 73.5% = +23.5 pp**
- Moderate behavioral necessity — partial flip rate over baseline. Inspect parse_fail to rule out incoherence.

## Reading guide
- `parse_fail_rate` should track between conditions; if it spikes under ablation while flip rate stays low, the heads matter for JSON formatting broadly (general damage), not FOLD-specific.
- Compare to `regenerate_ablated` block in `results/continuation_after_patch/{model}/SUMMARY.md` — should agree when `--pipeline recon --filter-recorded-bucket illegal_fold` is set.
