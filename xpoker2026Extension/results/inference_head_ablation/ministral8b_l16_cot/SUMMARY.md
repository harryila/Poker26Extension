# Inference-time head ablation (behavioral)

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- **Ablation scope:** zeros triplet/control heads at the **last sequence position on every forward pass during action `generate()`** (full CoT reasoning + JSON). This is **more aggressive** than single-position verb patching at L*.
- n_decisions: 200 (seed=42)
- Triplet heads: `[22, 9, 15]`
- Control heads: `[0, 1, 2]`

| Condition | illegal_FOLD rate | clean_CHECK rate | parse OK | fallback |
|---|---:|---:|---:|---:|
| **baseline** | 38.0% | 5.0% | 62.0% | 38.0% |
| **triplet** | 42.5% | 3.0% | 57.5% | 42.5% |
| **control** | 34.5% | 8.0% | 65.5% | 34.5% |

**Δ illegal_FOLD (triplet − baseline): +4.5 pp**
- Mixed or increased illegal_FOLD rate; interpret with per-row JSONL and recorded vs replay bucket columns.

## Interpreting necessity vs general generation damage
- If **parse_success** and **fallback_rate** stay stable under triplet ablation but **illegal_FOLD** drops → circuit-specific behavioral necessity for the failure mode.
- If **parse_success** / **fallback** degrade together with **illegal_FOLD** → heads matter for coherent generation broadly (true but less surgical than verb-position ablation).

## Reading guide
- Compare **replay_bucket** to **recorded_bucket** in `*_rows.jsonl` to see whether re-inference matches the original log under baseline.
- A large drop in `illegal_fold` under triplet ablation supports: the L* head set is necessary for the CoT-conditional failure mode.
