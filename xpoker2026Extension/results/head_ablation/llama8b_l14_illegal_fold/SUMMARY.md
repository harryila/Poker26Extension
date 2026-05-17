# Head ablation (necessity test) results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Target bucket: `illegal_fold`
- n_target: 50

Each row is one head set ablated (zeroed at the verb position).
`Δ(CHECK − FOLD)` is the change vs the baseline forward (negative = more FOLD-leaning under ablation).
`top-1 family changed` is the fraction of targets whose top-1 family (CHECK_CALL / FOLD / BET_RAISE / OTHER) changed under ablation. Higher = more disruptive.
`verb predicted (ablated)` is the fraction of targets whose ablated top-1 still matches the recorded verb. Lower = bigger necessity finding (heads matter for the verb prediction).

| Head set | n | mean Δ(CHECK − FOLD) | top-1 family changed | verb predicted (baseline → ablated) |
|---|---:|---:|---:|---:|
| `5+23+24` | 50 | +2.153 | **2.0%** | 98.0% → **98.0%** |
| `2+5+23+24` | 50 | +2.256 | **2.0%** | 98.0% → **96.0%** |
| `23` | 50 | +1.474 | **0.0%** | 98.0% → **98.0%** |
| `0+1+7+9` | 50 | -1.216 | **2.0%** | 98.0% → **100.0%** |

## Reading guide
- **Verb-predicted-baseline ≫ Verb-predicted-ablated**: the heads are necessary for the verb prediction. Strong necessity finding.
- **Top-1 family changed ≥ 50% AND mean Δ in expected sign**: ablation reliably shifts the model's verb-distribution toward the alternative family. Cite the specific heads.
- **A control set of random heads showing 0% family change AND near-zero Δ**: confirms the necessity finding is specific to the dominant heads, not generic to ablating any heads.
