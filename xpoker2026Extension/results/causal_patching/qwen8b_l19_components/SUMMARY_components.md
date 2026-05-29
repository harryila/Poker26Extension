# Component-level causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=24)
- Layer: **19**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=19
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 240 | +4.833 | +100% | 1.2% | 98.8% | 0.0% |
| `attn` | 240 | +3.770 | +78% | 0.4% | 99.6% | 0.0% |
| `mlp` | 240 | +0.869 | +18% | 0.0% | 100.0% | 0.0% |
| `head_00` | 240 | +0.523 | +11% | 0.0% | 100.0% | 0.0% |
| `head_01` | 240 | +0.669 | +14% | 0.0% | 100.0% | 0.0% |
| `head_02` | 240 | -0.340 | -7% | 0.0% | 100.0% | 0.0% |
| `head_03` | 240 | +0.722 | +15% | 0.0% | 100.0% | 0.0% |
| `head_04` | 240 | +0.005 | +0% | 0.0% | 100.0% | 0.0% |
| `head_05` | 240 | +0.016 | +0% | 0.0% | 100.0% | 0.0% |
| `head_06` | 240 | +0.024 | +0% | 0.0% | 100.0% | 0.0% |
| `head_07` | 240 | -0.023 | -0% | 0.0% | 100.0% | 0.0% |
| `head_08` | 240 | +0.010 | +0% | 0.0% | 100.0% | 0.0% |
| `head_09` | 240 | +0.035 | +1% | 0.0% | 100.0% | 0.0% |
| `head_10` | 240 | -0.124 | -3% | 0.0% | 100.0% | 0.0% |
| `head_11` | 240 | +0.042 | +1% | 0.0% | 100.0% | 0.0% |
| `head_12` | 240 | -0.437 | -9% | 0.0% | 100.0% | 0.0% |
| `head_13` | 240 | +0.370 | +8% | 0.0% | 100.0% | 0.0% |
| `head_14` | 240 | -0.248 | -5% | 0.0% | 100.0% | 0.0% |
| `head_15` | 240 | +0.065 | +1% | 0.0% | 100.0% | 0.0% |
| `head_16` | 240 | +0.010 | +0% | 0.0% | 100.0% | 0.0% |
| `head_17` | 240 | -0.034 | -1% | 0.0% | 100.0% | 0.0% |
| `head_18` | 240 | +0.057 | +1% | 0.0% | 100.0% | 0.0% |
| `head_19` | 240 | +0.035 | +1% | 0.0% | 100.0% | 0.0% |
| `head_20` | 240 | -0.044 | -1% | 0.0% | 100.0% | 0.0% |
| `head_21` | 240 | +0.741 | +15% | 0.0% | 100.0% | 0.0% |
| `head_22` | 240 | -0.158 | -3% | 0.0% | 100.0% | 0.0% |
| `head_23` | 240 | +0.215 | +4% | 0.0% | 100.0% | 0.0% |
| `head_24` | 240 | -0.068 | -1% | 0.0% | 100.0% | 0.0% |
| `head_25` | 240 | -0.016 | -0% | 0.0% | 100.0% | 0.0% |
| `head_26` | 240 | -0.021 | -0% | 0.0% | 100.0% | 0.0% |
| `head_27` | 240 | +0.051 | +1% | 0.0% | 100.0% | 0.0% |
| `head_28` | 240 | +0.161 | +3% | 0.0% | 100.0% | 0.0% |
| `head_29` | 240 | +0.526 | +11% | 0.0% | 100.0% | 0.0% |
| `head_30` | 240 | -0.208 | -4% | 0.0% | 100.0% | 0.0% |
| `head_31` | 240 | +0.822 | +17% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
