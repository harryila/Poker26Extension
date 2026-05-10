# Component-level causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched log: `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layer: **14**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 0.967
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=14
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 300 | +7.940 | +100% | 80.7% | 11.7% | 7.7% |
| `attn` | 300 | +3.872 | +49% | 24.0% | 75.3% | 0.7% |
| `mlp` | 300 | -0.599 | -8% | 0.3% | 93.0% | 6.7% |
| `head_00` | 300 | -0.153 | -2% | 3.3% | 96.3% | 0.3% |
| `head_01` | 300 | +0.179 | +2% | 0.3% | 94.3% | 5.3% |
| `head_02` | 300 | +0.713 | +9% | 3.3% | 94.3% | 2.3% |
| `head_03` | 300 | -0.195 | -2% | 3.3% | 95.3% | 1.3% |
| `head_04` | 300 | -0.075 | -1% | 2.7% | 97.0% | 0.3% |
| `head_05` | 300 | +1.374 | +17% | 3.3% | 96.7% | 0.0% |
| `head_06` | 300 | -0.331 | -4% | 0.7% | 93.7% | 5.7% |
| `head_07` | 300 | -0.010 | -0% | 2.0% | 98.0% | 0.0% |
| `head_08` | 300 | -0.027 | -0% | 3.3% | 96.7% | 0.0% |
| `head_09` | 300 | +0.034 | +0% | 3.3% | 96.7% | 0.0% |
| `head_10` | 300 | +0.069 | +1% | 3.3% | 96.0% | 0.7% |
| `head_11` | 300 | +0.150 | +2% | 3.3% | 96.7% | 0.0% |
| `head_12` | 300 | +0.045 | +1% | 3.3% | 96.7% | 0.0% |
| `head_13` | 300 | +0.167 | +2% | 3.3% | 95.7% | 1.0% |
| `head_14` | 300 | -0.074 | -1% | 3.3% | 96.7% | 0.0% |
| `head_15` | 300 | -0.009 | -0% | 3.3% | 96.7% | 0.0% |
| `head_16` | 300 | -0.093 | -1% | 3.3% | 96.7% | 0.0% |
| `head_17` | 300 | +0.033 | +0% | 3.3% | 96.7% | 0.0% |
| `head_18` | 300 | -0.041 | -1% | 3.3% | 96.7% | 0.0% |
| `head_19` | 300 | -0.005 | -0% | 3.3% | 96.7% | 0.0% |
| `head_20` | 300 | +0.309 | +4% | 3.3% | 96.7% | 0.0% |
| `head_21` | 300 | -0.526 | -7% | 3.0% | 96.7% | 0.3% |
| `head_22` | 300 | -0.354 | -4% | 1.0% | 94.0% | 5.0% |
| `head_23` | 300 | +2.887 | +36% | 7.3% | 92.3% | 0.3% |
| `head_24` | 300 | +1.577 | +20% | 7.3% | 91.3% | 1.3% |
| `head_25` | 300 | -0.038 | -0% | 3.3% | 96.7% | 0.0% |
| `head_26` | 300 | +0.085 | +1% | 3.3% | 96.3% | 0.3% |
| `head_27` | 300 | -0.243 | -3% | 1.0% | 94.0% | 5.0% |
| `head_28` | 300 | -0.444 | -6% | 3.3% | 96.7% | 0.0% |
| `head_29` | 300 | +0.223 | +3% | 3.3% | 96.0% | 0.7% |
| `head_30` | 300 | -0.058 | -1% | 3.3% | 96.7% | 0.0% |
| `head_31` | 300 | +0.498 | +6% | 3.3% | 95.7% | 1.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
