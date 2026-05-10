# Component-level causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=24)
- Layer: **23**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=23
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 240 | +26.831 | +100% | 100.0% | 0.0% | 0.0% |
| `attn` | 240 | +2.626 | +10% | 0.0% | 100.0% | 0.0% |
| `mlp` | 240 | +2.045 | +8% | 0.0% | 100.0% | 0.0% |
| `head_00` | 240 | +1.167 | +4% | 0.0% | 100.0% | 0.0% |
| `head_01` | 240 | -2.263 | -8% | 0.0% | 100.0% | 0.0% |
| `head_02` | 240 | +1.797 | +7% | 0.0% | 100.0% | 0.0% |
| `head_03` | 240 | +1.037 | +4% | 0.0% | 100.0% | 0.0% |
| `head_04` | 240 | +1.231 | +5% | 0.0% | 100.0% | 0.0% |
| `head_05` | 240 | +0.031 | +0% | 0.0% | 100.0% | 0.0% |
| `head_06` | 240 | -0.012 | -0% | 0.0% | 100.0% | 0.0% |
| `head_07` | 240 | +0.085 | +0% | 0.0% | 100.0% | 0.0% |
| `head_08` | 240 | +0.021 | +0% | 0.0% | 100.0% | 0.0% |
| `head_09` | 240 | -0.081 | -0% | 0.0% | 100.0% | 0.0% |
| `head_10` | 240 | +0.034 | +0% | 0.0% | 100.0% | 0.0% |
| `head_11` | 240 | +0.124 | +0% | 0.0% | 100.0% | 0.0% |
| `head_12` | 240 | +0.065 | +0% | 0.0% | 100.0% | 0.0% |
| `head_13` | 240 | +0.770 | +3% | 0.0% | 100.0% | 0.0% |
| `head_14` | 240 | +0.021 | +0% | 0.0% | 100.0% | 0.0% |
| `head_15` | 240 | +0.569 | +2% | 0.0% | 100.0% | 0.0% |
| `head_16` | 240 | +0.010 | +0% | 0.0% | 100.0% | 0.0% |
| `head_17` | 240 | -0.020 | -0% | 0.0% | 100.0% | 0.0% |
| `head_18` | 240 | -0.026 | -0% | 0.0% | 100.0% | 0.0% |
| `head_19` | 240 | +0.005 | +0% | 0.0% | 100.0% | 0.0% |
| `head_20` | 240 | +0.014 | +0% | 0.0% | 100.0% | 0.0% |
| `head_21` | 240 | +0.094 | +0% | 0.0% | 100.0% | 0.0% |
| `head_22` | 240 | +0.009 | +0% | 0.0% | 100.0% | 0.0% |
| `head_23` | 240 | -0.094 | -0% | 0.0% | 100.0% | 0.0% |
| `head_24` | 240 | -0.914 | -3% | 0.0% | 100.0% | 0.0% |
| `head_25` | 240 | +0.527 | +2% | 0.0% | 100.0% | 0.0% |
| `head_26` | 240 | +2.987 | +11% | 1.7% | 98.3% | 0.0% |
| `head_27` | 240 | -0.017 | -0% | 0.0% | 100.0% | 0.0% |
| `head_28` | 240 | -2.713 | -10% | 0.0% | 100.0% | 0.0% |
| `head_29` | 240 | -0.084 | -0% | 0.0% | 100.0% | 0.0% |
| `head_30` | 240 | +2.665 | +10% | 0.0% | 100.0% | 0.0% |
| `head_31` | 240 | -1.480 | -6% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
