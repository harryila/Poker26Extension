# Component-level causal patching results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Enriched logs (pooled, n=3):
  - `logs/cot_ministral8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_ministral8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
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
| `residual` | 300 | +7.808 | +100% | 100.0% | 0.0% | 0.0% |
| `attn` | 300 | +2.986 | +38% | 5.7% | 94.3% | 0.0% |
| `heads_09_15_22_24_30_31` | 300 | +4.316 | +55% | 37.0% | 63.0% | 0.0% |
| `mlp` | 300 | +1.022 | +13% | 0.0% | 100.0% | 0.0% |
| `head_09` | 300 | +0.651 | +8% | 0.0% | 100.0% | 0.0% |
| `head_15` | 300 | +0.492 | +6% | 0.0% | 100.0% | 0.0% |
| `head_22` | 300 | +2.390 | +31% | 0.0% | 100.0% | 0.0% |
| `head_24` | 300 | +0.217 | +3% | 0.0% | 100.0% | 0.0% |
| `head_30` | 300 | +0.265 | +3% | 0.0% | 100.0% | 0.0% |
| `head_31` | 300 | +0.211 | +3% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
