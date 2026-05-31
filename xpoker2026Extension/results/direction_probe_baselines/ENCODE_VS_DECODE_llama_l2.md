# Encode-vs-decode: is the Bayesian posterior decodable from the residual?

- Tagged residuals: `results/direction_probe/llama8b_l2/raw_residuals_tagged.npz`  (rows with oracle: 757)
- Target posterior: `oracle_strategy_aware`  (14 buckets; 'trash' index 13)
- Decodability = 5-fold CV R^2 of ridge regression residual → posterior probability.

## Per-bucket decodability (CV R^2)
| bucket | CV R^2 |
|---|---:|
| premium_pairs | +0.582 |
| strong_pairs | +0.575 |
| medium_pairs | +0.619 |
| small_pairs | +0.719 |
| premium_broadway | +0.557 |
| strong_broadway | +0.611 |
| medium_broadway | +0.646 |
| suited_aces | +0.553 |
| suited_connectors | +0.674 |
| suited_gappers | +0.668 |
| offsuit_connectors | +0.547 |
| weak_broadway | +0.423 |
| speculative_suited | +0.613 |
| trash | +0.635 |

- **trash-mass decodability R^2 = +0.635**  (mean over buckets +0.602)

## The encode-vs-decode gap (rows with a stated belief)
- mean **JS(stated belief, oracle) = 0.195 nats** (the decode-side miscalibration; higher = worse).
- oracle trash mass (truth): mean **0.659**
- model STATED trash mass: mean **0.659**  (|err| 0.322)
- PROBE-recovered trash from the residual: mean **0.657**  (|err| 0.025)

**Interpretation:** the residual probe recovers oracle trash mass MORE accurately than the model's own stated belief does. → the agent ENCODES more of the correct posterior than it STATES: miscalibration is (at least partly) a readout failure.

## Reading
- High oracle decodability (R^2) + high stated-belief JS = **knows-but-mis-states** (the citable Feng-et-al-style finding).
- The saved steering direction (residual→trash ridge weights) is used by `posterior_steering.py` to test whether ADDING it de-biases the stated belief / decision.
