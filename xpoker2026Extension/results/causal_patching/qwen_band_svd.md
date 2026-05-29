# Qwen band SVD — distributed but low-rank?

- model=Qwen/Qwen3-8B bucket=clean_check_or_call n=150

| layer | n | rank for 90% var | top-1 SV energy | cos(decision dir, top-3 SVs) |
|---:|---:|---:|---:|---|
| 18 | 150 | 24 | 32% | 0.07, 0.03, 0.05 |
| 19 | 150 | 25 | 30% | 0.13, 0.08, 0.02 |
| 20 | 150 | 24 | 30% | 0.13, 0.09, 0.04 |
| 23 | 150 | 18 | 34% | 0.11, 0.01, 0.11 |

## Reading
- Low `rank for 90% var` at L18-20 = the distributed-across-heads computation is nonetheless LOW-RANK in activation space (few directions carry it).
- High cos(decision dir, top SVs) = the learned verb direction lives in that low-rank band — reconciling 'distributed heads' with 'a single usable direction'.
- Follow-up: repeat at the o_proj-input (per-head) level to attribute the singular directions to head groups (Ahmad et al. 2025).
