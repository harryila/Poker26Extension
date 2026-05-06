# Tier 1A.small CoT + logit-lens — descriptive summary

Per-cell logit-lens descriptive stats (from `analyze_logit_lens.py`). Grid size depends on SEEDS / TEMPS env vars — default is 1 seed × 1 temp × 3 models = 3 cells. See script header for grid options.

| Cell | n records | Layers | Crystallization (mean) | (median) |
|---|---:|---:|---:|---:|
| `llama8b_t02_s123_informative_v2_logitlens` | 317 | 32 | 31.0 | 31.0 |
| `llama8b_t02_s42_informative_v2_logitlens` | 355 | 32 | 31.0 | 31.0 |
| `llama8b_t02_s456_informative_v2_logitlens` | 311 | 32 | 31.0 | 31.0 |
| `llama8b_t0_s123_informative_v2_logitlens` | 351 | 32 | 31.0 | 31.0 |
| `llama8b_t0_s42_informative_v2_logitlens` | 350 | 32 | 31.0 | 31.0 |
| `llama8b_t0_s456_informative_v2_logitlens` | 316 | 32 | 31.0 | 31.0 |
| `ministral8b_t02_s123_informative_v2_logitlens` | 106 | 36 | 35.0 | 35.0 |
| `ministral8b_t02_s42_informative_v2_logitlens` | 309 | 36 | 35.0 | 35.0 |
| `ministral8b_t02_s456_informative_v2_logitlens` | 108 | 36 | 35.0 | 35.0 |
| `ministral8b_t0_s123_informative_v2_logitlens` | 102 | 36 | 35.0 | 35.0 |
| `ministral8b_t0_s42_informative_v2_logitlens` | 313 | 36 | 35.0 | 35.0 |
| `ministral8b_t0_s456_informative_v2_logitlens` | 106 | 36 | 35.0 | 35.0 |
| `qwen8b_t02_s123_informative_v2_logitlens` | 394 | 36 | 35.0 | 35.0 |
| `qwen8b_t02_s42_informative_v2_logitlens` | 460 | 36 | 35.0 | 35.0 |
| `qwen8b_t02_s456_informative_v2_logitlens` | 330 | 36 | 35.0 | 35.0 |
| `qwen8b_t0_s123_informative_v2_logitlens` | 350 | 36 | 35.0 | 35.0 |
| `qwen8b_t0_s42_informative_v2_logitlens` | 455 | 36 | 35.0 | 35.0 |
| `qwen8b_t0_s456_informative_v2_logitlens` | 357 | 36 | 35.0 | 35.0 |

*Crystallization layer* = earliest layer from which the top-1 token never changes through to the final layer. Lower = the model 'decided' earlier. Higher = the model is still revising late.

## Companion artifacts

- `entropy_<cell>.png` — per-layer mean entropy plot for each cell.
- `logitlens_<cell>.json` — full per-cell stats from `analyze_logit_lens.py`.
- `by_failure_mode_<cell>.json` + `BY_FAILURE_MODE.md` —
  the mechanistic answer: per-bucket (clean / illegal_fold / ...)
  per-layer mapped action group at the action-emission token.
- `logs/cot_<cell>_logit_lens.jsonl` — raw sidecar (per-decision
  per-layer top-1 tokens + entropies).

## Mechanistic question this run targets

> Do hidden states encode `CHECK_OR_CALL` even when the model verbalizes
> `FOLD` into a free-check spot?

That question is answered by Phase 3b's `BY_FAILURE_MODE.md` — see the
`illegal_fold` bucket per cell. If early layers favour CHECK while only
the final layers cross to FOLD, that's a verbalization-stage failure;
if FOLD dominates from layer 0, the model is FOLD-committed top to bottom.
