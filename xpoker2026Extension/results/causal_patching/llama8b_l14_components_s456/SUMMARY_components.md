# Component-level causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched log: `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=18)
- Layer: **14**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=14
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 180 | +6.942 | +100% | 73.9% | 25.0% | 1.1% |
| `attn` | 180 | +3.296 | +47% | 23.3% | 76.7% | 0.0% |
| `mlp` | 180 | -0.488 | -7% | 0.0% | 98.3% | 1.7% |
| `head_00` | 180 | -0.163 | -2% | 0.0% | 100.0% | 0.0% |
| `head_01` | 180 | +0.130 | +2% | 5.6% | 94.4% | 0.0% |
| `head_02` | 180 | +0.638 | +9% | 5.0% | 95.0% | 0.0% |
| `head_03` | 180 | -0.175 | -3% | 0.0% | 100.0% | 0.0% |
| `head_04` | 180 | -0.050 | -1% | 1.1% | 98.9% | 0.0% |
| `head_05` | 180 | +1.303 | +19% | 4.4% | 95.6% | 0.0% |
| `head_06` | 180 | -0.260 | -4% | 0.0% | 100.0% | 0.0% |
| `head_07` | 180 | -0.169 | -2% | 0.0% | 100.0% | 0.0% |
| `head_08` | 180 | -0.012 | -0% | 0.0% | 100.0% | 0.0% |
| `head_09` | 180 | +0.011 | +0% | 0.0% | 100.0% | 0.0% |
| `head_10` | 180 | +0.069 | +1% | 0.6% | 99.4% | 0.0% |
| `head_11` | 180 | +0.149 | +2% | 0.0% | 100.0% | 0.0% |
| `head_12` | 180 | +0.024 | +0% | 0.0% | 100.0% | 0.0% |
| `head_13` | 180 | +0.128 | +2% | 0.0% | 100.0% | 0.0% |
| `head_14` | 180 | -0.062 | -1% | 0.0% | 100.0% | 0.0% |
| `head_15` | 180 | -0.007 | -0% | 0.0% | 100.0% | 0.0% |
| `head_16` | 180 | -0.116 | -2% | 0.0% | 100.0% | 0.0% |
| `head_17` | 180 | +0.047 | +1% | 0.0% | 100.0% | 0.0% |
| `head_18` | 180 | -0.044 | -1% | 0.0% | 100.0% | 0.0% |
| `head_19` | 180 | -0.036 | -1% | 0.0% | 100.0% | 0.0% |
| `head_20` | 180 | +0.214 | +3% | 0.6% | 99.4% | 0.0% |
| `head_21` | 180 | -0.593 | -9% | 0.0% | 100.0% | 0.0% |
| `head_22` | 180 | -0.224 | -3% | 0.0% | 100.0% | 0.0% |
| `head_23` | 180 | +2.410 | +35% | 5.0% | 95.0% | 0.0% |
| `head_24` | 180 | +1.337 | +19% | 5.6% | 94.4% | 0.0% |
| `head_25` | 180 | -0.050 | -1% | 0.0% | 100.0% | 0.0% |
| `head_26` | 180 | +0.083 | +1% | 0.0% | 100.0% | 0.0% |
| `head_27` | 180 | -0.240 | -3% | 0.0% | 100.0% | 0.0% |
| `head_28` | 180 | -0.393 | -6% | 0.0% | 100.0% | 0.0% |
| `head_29` | 180 | +0.190 | +3% | 0.0% | 100.0% | 0.0% |
| `head_30` | 180 | -0.066 | -1% | 1.1% | 98.9% | 0.0% |
| `head_31` | 180 | +0.391 | +6% | 1.1% | 98.9% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
