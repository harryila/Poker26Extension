# Sufficiency with target-clustered CIs — layer L23

- Pooled cell: `results/causal_patching/qwen8b_t0_pooled_layer_sweep`
- Pairs at L23: 240 (= n_source × n_target); **independent units (target decisions): 24**
- self-check: recomputed mean Δ = +26.831 vs summary.json +26.831 -> OK
- Bootstrap resamples TARGET decisions (the independent unit), not pairs. 95% percentile CI.

## Pooled, clustered by target
- **top-1 → CHECK: 100.0%**  (95% CI [100.0, 100.0])
- **mean Δ(CHECK−FOLD): +26.83 nats**  (95% CI [+26.04, +27.56])
- random-source null at L23: +8.50 nats  → **spec-adj Δ ≈ +18.33** (CI [+17.54, +19.06], null treated as fixed)

## Per-seed spread (magnitude is seed-sensitive — report the range)
| seed cell | n_target | top-1 → CHECK | mean Δ | null | spec-adj |
|---|---:|---:|---:|---:|---:|
| qwen8b_t0_s42_replicate | 4 | 100.0% | +28.70 | +3.65 | +25.05 |
| qwen8b_t0_s123_replicate | 9 | 100.0% | +32.26 | +16.70 | +15.56 |
| qwen8b_t0_s456_replicate | 11 | 100.0% | +28.14 | +3.20 | +24.94 |

- **spec-adj across seeds: mean +21.85, range [+15.56, +25.05]** (report this range, not the pooled point — the null varies ~5× by seed).

## Reading
- The categorical sufficiency (top-1 → CHECK) is the robust headline; if its CI is [100,100] it means every target flips under every source (saturated).
- The nat-magnitude spec-adj is seed-sensitive; cite the per-seed range and the target-clustered CI, NOT the single pooled value.
