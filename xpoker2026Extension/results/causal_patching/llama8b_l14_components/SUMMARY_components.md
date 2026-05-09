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
| `mlp` | 300 | -0.500 | -6% | 0.0% | 100.0% | 0.0% |
| `head_00` | 300 | -0.176 | -2% | 0.0% | 100.0% | 0.0% |
| `head_01` | 300 | +0.120 | +2% | 0.0% | 98.0% | 2.0% |
| `head_02` | 300 | +0.797 | +10% | 0.0% | 100.0% | 0.0% |
| `head_03` | 300 | -0.191 | -2% | 0.0% | 100.0% | 0.0% |
| `head_04` | 300 | -0.103 | -1% | 0.0% | 100.0% | 0.0% |
| `head_05` | 300 | +1.396 | +18% | 0.0% | 100.0% | 0.0% |
| `head_06` | 300 | -0.384 | -5% | 0.0% | 100.0% | 0.0% |
| `head_07` | 300 | -0.041 | -1% | 0.0% | 100.0% | 0.0% |
| `head_08` | 300 | -0.011 | -0% | 0.0% | 100.0% | 0.0% |
| `head_09` | 300 | +0.035 | +0% | 0.0% | 100.0% | 0.0% |
| `head_10` | 300 | +0.085 | +1% | 0.0% | 100.0% | 0.0% |
| `head_11` | 300 | +0.150 | +2% | 0.0% | 100.0% | 0.0% |
| `head_12` | 300 | +0.042 | +1% | 0.0% | 100.0% | 0.0% |
| `head_13` | 300 | +0.146 | +2% | 0.0% | 100.0% | 0.0% |
| `head_14` | 300 | -0.071 | -1% | 0.0% | 100.0% | 0.0% |
| `head_15` | 300 | -0.018 | -0% | 0.0% | 100.0% | 0.0% |
| `head_16` | 300 | -0.126 | -2% | 0.0% | 100.0% | 0.0% |
| `head_17` | 300 | +0.055 | +1% | 0.0% | 100.0% | 0.0% |
| `head_18` | 300 | -0.053 | -1% | 0.0% | 100.0% | 0.0% |
| `head_19` | 300 | -0.031 | -0% | 0.0% | 100.0% | 0.0% |
| `head_20` | 300 | +0.276 | +3% | 0.0% | 100.0% | 0.0% |
| `head_21` | 300 | -0.549 | -7% | 0.0% | 100.0% | 0.0% |
| `head_22` | 300 | -0.238 | -3% | 0.0% | 99.3% | 0.7% |
| `head_23` | 300 | +2.726 | +35% | 0.7% | 99.3% | 0.0% |
| `head_24` | 300 | +1.619 | +20% | 3.7% | 96.3% | 0.0% |
| `head_25` | 300 | -0.050 | -1% | 0.0% | 100.0% | 0.0% |
| `head_26` | 300 | +0.080 | +1% | 0.0% | 100.0% | 0.0% |
| `head_27` | 300 | -0.299 | -4% | 0.0% | 99.7% | 0.3% |
| `head_28` | 300 | -0.416 | -5% | 0.0% | 100.0% | 0.0% |
| `head_29` | 300 | +0.238 | +3% | 0.0% | 100.0% | 0.0% |
| `head_30` | 300 | -0.082 | -1% | 0.0% | 100.0% | 0.0% |
| `head_31` | 300 | +0.473 | +6% | 0.0% | 100.0% | 0.0% |

**Interpretation guide**:
- `residual` is the existing experiment baseline (full-residual patch); expected to match prior pooled-sweep numbers at this layer.
- `attn` ≈ `residual` (≥80% ratio) and `mlp` ≪ `residual` (<20%): the layer's effect is mediated by attention, not the MLP.
- A small set of `head_NN` rows each contributing >10% of the residual effect: SPARSE HEAD STORY — the strongest possible version of the result. Cite those head indices.
- All `head_NN` rows roughly equal and individually small: dense attention story — the effect spreads across many heads and no single head dominates.
- `attn` ≈ `residual` AND no head individually large: a weighted combination of heads carries the signal (not as strong a result, still publishable).
- `attn` < `residual` AND `mlp` < `residual` AND their sum ≠ `residual`: a real interaction between sublayers exists; both are needed to flip the verb.
