# Component-level causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched logs (pooled, n=3):
  - `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layer: **17**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=17
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 300 | +8.750 | +100% | 100.0% | 0.0% | 0.0% |
| `attn` | 300 | +2.678 | +31% | 0.0% | 100.0% | 0.0% |
| `mlp` | 300 | +0.172 | +2% | 0.0% | 100.0% | 0.0% |
| `head_00` | 300 | +0.011 | +0% | 0.0% | 100.0% | 0.0% |
| `head_01` | 300 | +0.001 | +0% | 0.0% | 100.0% | 0.0% |
| `head_02` | 300 | +0.001 | +0% | 0.0% | 100.0% | 0.0% |
| `head_03` | 300 | +0.009 | +0% | 0.0% | 100.0% | 0.0% |
| `head_04` | 300 | +0.286 | +3% | 0.0% | 100.0% | 0.0% |
| `head_05` | 300 | +0.070 | +1% | 0.0% | 100.0% | 0.0% |
| `head_06` | 300 | -0.358 | -4% | 0.0% | 100.0% | 0.0% |
| `head_07` | 300 | +0.404 | +5% | 0.0% | 100.0% | 0.0% |
| `head_08` | 300 | -0.121 | -1% | 0.0% | 100.0% | 0.0% |
| `head_09` | 300 | +0.615 | +7% | 0.0% | 100.0% | 0.0% |
| `head_10` | 300 | +0.046 | +1% | 0.0% | 100.0% | 0.0% |
| `head_11` | 300 | +0.050 | +1% | 0.0% | 100.0% | 0.0% |
| `head_12` | 300 | +0.084 | +1% | 0.0% | 100.0% | 0.0% |
| `head_13` | 300 | +0.338 | +4% | 0.0% | 100.0% | 0.0% |
| `head_14` | 300 | -0.405 | -5% | 0.0% | 100.0% | 0.0% |
| `head_15` | 300 | -0.066 | -1% | 0.0% | 100.0% | 0.0% |
| `head_16` | 300 | +0.006 | +0% | 0.0% | 100.0% | 0.0% |
| `head_17` | 300 | +0.006 | +0% | 0.0% | 100.0% | 0.0% |
| `head_18` | 300 | +0.019 | +0% | 0.0% | 100.0% | 0.0% |
| `head_19` | 300 | -0.013 | -0% | 0.0% | 100.0% | 0.0% |
| `head_20` | 300 | +0.002 | +0% | 0.0% | 100.0% | 0.0% |
| `head_21` | 300 | +0.031 | +0% | 0.0% | 100.0% | 0.0% |
| `head_22` | 300 | -0.017 | -0% | 0.0% | 100.0% | 0.0% |
| `head_23` | 300 | +0.001 | +0% | 0.0% | 100.0% | 0.0% |
| `head_24` | 300 | +0.035 | +0% | 0.0% | 100.0% | 0.0% |
| `head_25` | 300 | +0.044 | +1% | 0.0% | 100.0% | 0.0% |
| `head_26` | 300 | +0.074 | +1% | 0.0% | 100.0% | 0.0% |
| `head_27` | 300 | +0.035 | +0% | 0.0% | 100.0% | 0.0% |
| `head_28` | 300 | +1.281 | +15% | 0.0% | 100.0% | 0.0% |
| `head_29` | 300 | -0.395 | -5% | 0.0% | 100.0% | 0.0% |
| `head_30` | 300 | +0.137 | +2% | 0.0% | 100.0% | 0.0% |
| `head_31` | 300 | +1.143 | +13% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
