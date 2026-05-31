# Encode-vs-decode: is the Bayesian posterior decodable from the residual?

- Tagged residuals: `results/direction_probe/qwen8b_l2/raw_residuals_tagged.npz`  (rows with oracle: 685)
- Target posterior: `oracle_strategy_aware`  (14 buckets; 'trash' index 13)
- Decodability = 5-fold CV R^2 of ridge regression residual → posterior probability.

## Per-bucket decodability (CV R^2)
| bucket | CV R^2 |
|---|---:|
| premium_pairs | +0.381 |
| strong_pairs | +0.325 |
| medium_pairs | +0.385 |
| small_pairs | +0.538 |
| premium_broadway | +0.365 |
| strong_broadway | +0.370 |
| medium_broadway | +0.343 |
| suited_aces | +0.349 |
| suited_connectors | +0.422 |
| suited_gappers | +0.417 |
| offsuit_connectors | +0.304 |
| weak_broadway | +0.319 |
| speculative_suited | +0.474 |
| trash | +0.339 |

- **trash-mass decodability R^2 = +0.339**  (mean over buckets +0.381)

## The encode-vs-decode gap (rows with a stated belief)
- mean **JS(stated belief, oracle) = 0.246 nats** (the decode-side miscalibration; higher = worse).
- oracle trash mass (truth): mean **0.664**
- model STATED trash mass: mean **0.110**  (|err| 0.556)
- PROBE-recovered trash from the residual: mean **0.664**  (|err| 0.034)

**Interpretation:** the residual probe recovers oracle trash mass MORE accurately than the model's own stated belief does. → the agent ENCODES more of the correct posterior than it STATES: miscalibration is (at least partly) a readout failure.

## Reading
- High oracle decodability (R^2) + high stated-belief JS = **knows-but-mis-states** (the citable Feng-et-al-style finding).
- The saved steering direction (residual→trash ridge weights) is used by `posterior_steering.py` to test whether ADDING it de-biases the stated belief / decision.
