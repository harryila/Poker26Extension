# Component-level causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched log: `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=16)
- Layer: **14**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=14
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 160 | +8.365 | +100% | 79.4% | 16.9% | 3.8% |
| `attn` | 160 | +3.889 | +46% | 10.6% | 89.4% | 0.0% |
| `mlp` | 160 | -0.558 | -7% | 0.0% | 97.5% | 2.5% |
| `head_00` | 160 | -0.172 | -2% | 0.0% | 100.0% | 0.0% |
| `head_01` | 160 | +0.143 | +2% | 0.0% | 97.5% | 2.5% |
| `head_02` | 160 | +0.821 | +10% | 0.0% | 100.0% | 0.0% |
| `head_03` | 160 | -0.204 | -2% | 0.0% | 100.0% | 0.0% |
| `head_04` | 160 | +0.145 | +2% | 0.0% | 100.0% | 0.0% |
| `head_05` | 160 | +1.289 | +15% | 0.0% | 100.0% | 0.0% |
| `head_06` | 160 | -0.362 | -4% | 0.0% | 100.0% | 0.0% |
| `head_07` | 160 | +0.112 | +1% | 0.0% | 100.0% | 0.0% |
| `head_08` | 160 | -0.002 | -0% | 0.0% | 100.0% | 0.0% |
| `head_09` | 160 | +0.025 | +0% | 0.0% | 100.0% | 0.0% |
| `head_10` | 160 | +0.046 | +1% | 0.0% | 100.0% | 0.0% |
| `head_11` | 160 | +0.150 | +2% | 0.0% | 100.0% | 0.0% |
| `head_12` | 160 | +0.055 | +1% | 0.0% | 100.0% | 0.0% |
| `head_13` | 160 | +0.161 | +2% | 0.0% | 100.0% | 0.0% |
| `head_14` | 160 | -0.106 | -1% | 0.0% | 100.0% | 0.0% |
| `head_15` | 160 | +0.010 | +0% | 0.0% | 100.0% | 0.0% |
| `head_16` | 160 | -0.129 | -2% | 0.0% | 100.0% | 0.0% |
| `head_17` | 160 | +0.004 | +0% | 0.0% | 100.0% | 0.0% |
| `head_18` | 160 | -0.038 | -0% | 0.0% | 100.0% | 0.0% |
| `head_19` | 160 | -0.023 | -0% | 0.0% | 100.0% | 0.0% |
| `head_20` | 160 | +0.320 | +4% | 0.0% | 100.0% | 0.0% |
| `head_21` | 160 | -0.472 | -6% | 0.0% | 100.0% | 0.0% |
| `head_22` | 160 | -0.387 | -5% | 0.0% | 98.8% | 1.2% |
| `head_23` | 160 | +2.877 | +34% | 1.2% | 98.8% | 0.0% |
| `head_24` | 160 | +1.638 | +20% | 5.0% | 95.0% | 0.0% |
| `head_25` | 160 | -0.043 | -1% | 0.0% | 100.0% | 0.0% |
| `head_26` | 160 | +0.102 | +1% | 0.0% | 100.0% | 0.0% |
| `head_27` | 160 | -0.258 | -3% | 0.0% | 100.0% | 0.0% |
| `head_28` | 160 | -0.457 | -5% | 0.0% | 100.0% | 0.0% |
| `head_29` | 160 | +0.252 | +3% | 0.0% | 100.0% | 0.0% |
| `head_30` | 160 | -0.132 | -2% | 0.0% | 100.0% | 0.0% |
| `head_31` | 160 | +0.551 | +7% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
