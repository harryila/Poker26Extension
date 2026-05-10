# Component-level causal patching results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Enriched logs (pooled, n=3):
  - `logs/cot_llama8b_t0_s42_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s123_informative_v2_logitlens_enriched.jsonl.gz`
  - `logs/cot_llama8b_t0_s456_informative_v2_logitlens_enriched.jsonl.gz`
- Source bucket: `clean_check_or_call` (n=10)
- Target bucket: `illegal_fold` (n=30)
- Layer: **14**
- Head geometry: num_heads=32, head_dim=128

## Controls
- `baseline_top1_match_rate` = 1.000
- `self_patch_max_logit_drift` = 0.000000

## Per-component / per-head effect at L=14
| Mode | n | mean Δ(CHECK − FOLD) | ratio to residual | top-1 → CHECK | top-1 → FOLD | top-1 → BET_RAISE |
|---|---:|---:|---:|---:|---:|---:|
| `residual` | 300 | +7.899 | +100% | 79.0% | 21.0% | 0.0% |
| `attn` | 300 | +3.846 | +49% | 14.3% | 85.7% | 0.0% |
| `heads_02_05_23_24` | 300 | +6.100 | +77% | 69.3% | 30.0% | 0.7% |
| `mlp` | 300 | -0.500 | -6% | 0.0% | 100.0% | 0.0% |
| `head_02` | 300 | +0.797 | +10% | 0.0% | 100.0% | 0.0% |
| `head_05` | 300 | +1.396 | +18% | 0.0% | 100.0% | 0.0% |
| `head_23` | 300 | +2.726 | +35% | 0.7% | 99.3% | 0.0% |
| `head_24` | 300 | +1.619 | +20% | 3.7% | 96.3% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- A `heads_NN_MM_KK` row near `attn`'s ratio: the listed sparse subset jointly captures the attention contribution. If its top-1 flip rate also matches `residual`, the triplet is the circuit.
- A `heads_NN_MM_KK` row well below the linear sum of its members' individual ratios: per-head contributions are NON-additive (the heads interact through downstream MLP recomputation).
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
