# Component-level causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layer: **15**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=15
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 300 | +11.661 | +100% | 100.0% | 0.0% | 0.0% |
| `attn` | 300 | +2.014 | +17% | 3.3% | 93.3% | 3.3% |
| `mlp` | 300 | +1.041 | +9% | 0.3% | 98.0% | 1.7% |
| `head_00` | 300 | +0.005 | +0% | 0.0% | 100.0% | 0.0% |
| `head_01` | 300 | +0.110 | +1% | 0.0% | 100.0% | 0.0% |
| `head_02` | 300 | -0.012 | -0% | 0.0% | 100.0% | 0.0% |
| `head_03` | 300 | +0.043 | +0% | 0.0% | 100.0% | 0.0% |
| `head_04` | 300 | +0.767 | +7% | 0.0% | 99.7% | 0.3% |
| `head_05` | 300 | +0.318 | +3% | 0.0% | 100.0% | 0.0% |
| `head_06` | 300 | -0.705 | -6% | 0.0% | 100.0% | 0.0% |
| `head_07` | 300 | +0.442 | +4% | 0.0% | 100.0% | 0.0% |
| `head_08` | 300 | +2.161 | +19% | 3.3% | 96.0% | 0.7% |
| `head_09` | 300 | +0.431 | +4% | 0.0% | 100.0% | 0.0% |
| `head_10` | 300 | -1.402 | -12% | 0.0% | 96.7% | 3.3% |
| `head_11` | 300 | +0.448 | +4% | 0.0% | 100.0% | 0.0% |
| `head_12` | 300 | -0.005 | -0% | 0.0% | 100.0% | 0.0% |
| `head_13` | 300 | -0.015 | -0% | 0.0% | 100.0% | 0.0% |
| `head_14` | 300 | +0.113 | +1% | 0.0% | 100.0% | 0.0% |
| `head_15` | 300 | -0.050 | -0% | 0.0% | 100.0% | 0.0% |
| `head_16` | 300 | -0.082 | -1% | 0.0% | 100.0% | 0.0% |
| `head_17` | 300 | +0.022 | +0% | 0.0% | 100.0% | 0.0% |
| `head_18` | 300 | +0.034 | +0% | 0.0% | 100.0% | 0.0% |
| `head_19` | 300 | -0.001 | -0% | 0.0% | 100.0% | 0.0% |
| `head_20` | 300 | -0.175 | -2% | 0.0% | 98.0% | 2.0% |
| `head_21` | 300 | +0.105 | +1% | 0.0% | 100.0% | 0.0% |
| `head_22` | 300 | +0.011 | +0% | 0.0% | 100.0% | 0.0% |
| `head_23` | 300 | +0.154 | +1% | 0.0% | 100.0% | 0.0% |
| `head_24` | 300 | -0.065 | -1% | 0.0% | 100.0% | 0.0% |
| `head_25` | 300 | -0.039 | -0% | 0.0% | 100.0% | 0.0% |
| `head_26` | 300 | +0.046 | +0% | 0.0% | 100.0% | 0.0% |
| `head_27` | 300 | +0.251 | +2% | 0.0% | 100.0% | 0.0% |
| `head_28` | 300 | +0.008 | +0% | 0.0% | 100.0% | 0.0% |
| `head_29` | 300 | +0.148 | +1% | 0.0% | 100.0% | 0.0% |
| `head_30` | 300 | +0.251 | +2% | 0.0% | 100.0% | 0.0% |
| `head_31` | 300 | +0.010 | +0% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
