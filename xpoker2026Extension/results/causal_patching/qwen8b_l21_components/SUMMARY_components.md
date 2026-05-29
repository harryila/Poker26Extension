# Component-level causal patching results

- Model: `Qwen/Qwen3-8B`
- Enriched logs (pooled, n=3):
  - `logs/cot_qwen8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_qwen8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=24)
- Layer: **21**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=21
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 240 | +13.246 | +100% | 55.8% | 44.2% | 0.0% |
| `attn` | 240 | +0.139 | +1% | 0.0% | 100.0% | 0.0% |
| `mlp` | 240 | -2.270 | -17% | 0.0% | 100.0% | 0.0% |
| `head_00` | 240 | -0.034 | -0% | 0.0% | 100.0% | 0.0% |
| `head_01` | 240 | -0.349 | -3% | 0.0% | 100.0% | 0.0% |
| `head_02` | 240 | -0.056 | -0% | 0.0% | 100.0% | 0.0% |
| `head_03` | 240 | -0.002 | -0% | 0.0% | 100.0% | 0.0% |
| `head_04` | 240 | -0.089 | -1% | 0.0% | 100.0% | 0.0% |
| `head_05` | 240 | -0.008 | -0% | 0.0% | 100.0% | 0.0% |
| `head_06` | 240 | +0.034 | +0% | 0.0% | 100.0% | 0.0% |
| `head_07` | 240 | -0.006 | -0% | 0.0% | 100.0% | 0.0% |
| `head_08` | 240 | +2.360 | +18% | 0.0% | 100.0% | 0.0% |
| `head_09` | 240 | -0.857 | -6% | 0.0% | 100.0% | 0.0% |
| `head_10` | 240 | -1.042 | -8% | 0.0% | 100.0% | 0.0% |
| `head_11` | 240 | +3.519 | +27% | 0.0% | 100.0% | 0.0% |
| `head_12` | 240 | -0.037 | -0% | 0.0% | 100.0% | 0.0% |
| `head_13` | 240 | +0.027 | +0% | 0.0% | 100.0% | 0.0% |
| `head_14` | 240 | -0.069 | -1% | 0.0% | 100.0% | 0.0% |
| `head_15` | 240 | +0.148 | +1% | 0.0% | 100.0% | 0.0% |
| `head_16` | 240 | +0.064 | +0% | 0.0% | 100.0% | 0.0% |
| `head_17` | 240 | -0.687 | -5% | 0.0% | 100.0% | 0.0% |
| `head_18` | 240 | +2.142 | +16% | 0.0% | 100.0% | 0.0% |
| `head_19` | 240 | +2.525 | +19% | 0.0% | 100.0% | 0.0% |
| `head_20` | 240 | +0.056 | +0% | 0.0% | 100.0% | 0.0% |
| `head_21` | 240 | +0.034 | +0% | 0.0% | 100.0% | 0.0% |
| `head_22` | 240 | -0.192 | -1% | 0.0% | 100.0% | 0.0% |
| `head_23` | 240 | +0.337 | +3% | 0.0% | 100.0% | 0.0% |
| `head_24` | 240 | +0.067 | +1% | 0.0% | 100.0% | 0.0% |
| `head_25` | 240 | +1.179 | +9% | 0.0% | 100.0% | 0.0% |
| `head_26` | 240 | -3.660 | -28% | 0.0% | 100.0% | 0.0% |
| `head_27` | 240 | +1.599 | +12% | 0.0% | 100.0% | 0.0% |
| `head_28` | 240 | -0.143 | -1% | 0.0% | 100.0% | 0.0% |
| `head_29` | 240 | -0.906 | -7% | 0.0% | 100.0% | 0.0% |
| `head_30` | 240 | +1.522 | +11% | 0.0% | 100.0% | 0.0% |
| `head_31` | 240 | -3.293 | -25% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
