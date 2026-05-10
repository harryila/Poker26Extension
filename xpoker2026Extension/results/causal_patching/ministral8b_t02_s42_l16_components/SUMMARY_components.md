# Component-level causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched log: `logs/cot_ministral8b_t02_s42_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layer: **16**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=16
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 300 | +6.834 | +100% | 87.3% | 12.7% | 0.0% |
| `attn` | 300 | +2.259 | +33% | 2.3% | 97.7% | 0.0% |
| `mlp` | 300 | +0.981 | +14% | 0.0% | 100.0% | 0.0% |
| `head_00` | 300 | -0.029 | -0% | 0.0% | 100.0% | 0.0% |
| `head_01` | 300 | +0.040 | +1% | 0.0% | 100.0% | 0.0% |
| `head_02` | 300 | +0.170 | +2% | 0.0% | 100.0% | 0.0% |
| `head_03` | 300 | +0.009 | +0% | 0.0% | 100.0% | 0.0% |
| `head_04` | 300 | +0.116 | +2% | 0.0% | 100.0% | 0.0% |
| `head_05` | 300 | +0.076 | +1% | 0.0% | 100.0% | 0.0% |
| `head_06` | 300 | +0.166 | +2% | 0.0% | 100.0% | 0.0% |
| `head_07` | 300 | -0.005 | -0% | 0.0% | 100.0% | 0.0% |
| `head_08` | 300 | -0.115 | -2% | 0.0% | 100.0% | 0.0% |
| `head_09` | 300 | +0.542 | +8% | 0.0% | 100.0% | 0.0% |
| `head_10` | 300 | +0.129 | +2% | 0.0% | 100.0% | 0.0% |
| `head_11` | 300 | -0.747 | -11% | 0.0% | 100.0% | 0.0% |
| `head_12` | 300 | +0.094 | +1% | 0.0% | 100.0% | 0.0% |
| `head_13` | 300 | -0.368 | -5% | 0.0% | 100.0% | 0.0% |
| `head_14` | 300 | -0.004 | -0% | 0.0% | 100.0% | 0.0% |
| `head_15` | 300 | +0.456 | +7% | 0.0% | 100.0% | 0.0% |
| `head_16` | 300 | +0.088 | +1% | 0.0% | 100.0% | 0.0% |
| `head_17` | 300 | +0.051 | +1% | 0.0% | 100.0% | 0.0% |
| `head_18` | 300 | +0.022 | +0% | 0.0% | 100.0% | 0.0% |
| `head_19` | 300 | +0.032 | +0% | 0.0% | 100.0% | 0.0% |
| `head_20` | 300 | +0.090 | +1% | 0.0% | 100.0% | 0.0% |
| `head_21` | 300 | -0.124 | -2% | 0.0% | 100.0% | 0.0% |
| `head_22` | 300 | +1.997 | +29% | 0.0% | 100.0% | 0.0% |
| `head_23` | 300 | -0.284 | -4% | 0.0% | 100.0% | 0.0% |
| `head_24` | 300 | +0.219 | +3% | 0.0% | 100.0% | 0.0% |
| `head_25` | 300 | -0.296 | -4% | 0.0% | 100.0% | 0.0% |
| `head_26` | 300 | +0.008 | +0% | 0.0% | 100.0% | 0.0% |
| `head_27` | 300 | +0.031 | +0% | 0.0% | 100.0% | 0.0% |
| `head_28` | 300 | +0.031 | +0% | 0.0% | 100.0% | 0.0% |
| `head_29` | 300 | +0.112 | +2% | 0.0% | 100.0% | 0.0% |
| `head_30` | 300 | +0.312 | +5% | 0.0% | 100.0% | 0.0% |
| `head_31` | 300 | +0.153 | +2% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
