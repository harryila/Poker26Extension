# Steer-to-repair-belief — does steering reduce JS(stated belief, oracle)?

- model=Qwen/Qwen3-8B layer=19 target=None n=40
- JS lower = belief closer to the Bayesian oracle. parse_rate guards against over-steer.

| alpha | trash: mean JS | trash: parse | control: mean JS | control: parse |
|---:|---:|---:|---:|---:|
| 0.0 | 0.250 | 100% | 0.250 | 100% |
| 2.0 | — | 0% | — | 0% |
| 4.0 | — | 0% | — | 0% |
| 6.0 | — | 0% | — | 0% |

## Reading
- trash JS DROPS with alpha while parse_rate stays high, AND drops more than the random control ⇒ steering causally repairs the readout (the belief moves toward the posterior the residual already encodes — the intervention converse of encode_vs_decode).
- If parse_rate collapses, alpha is too large (broken generation) — lower it.
