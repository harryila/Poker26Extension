# Encode-vs-decode: is the Bayesian posterior decodable from the residual?

- Tagged residuals: `results/direction_probe/ministral8b_l2/raw_residuals_tagged.npz`  (rows with oracle: 514)
- Target posterior: `oracle_strategy_aware`  (14 buckets; 'trash' index 13)
- Decodability = 5-fold CV R^2 of ridge regression residual → posterior probability.

## Per-bucket decodability (CV R^2)
| bucket | CV R^2 |
|---|---:|
| premium_pairs | +0.891 |
| strong_pairs | +0.858 |
| medium_pairs | +0.822 |
| small_pairs | +0.790 |
| premium_broadway | +0.854 |
| strong_broadway | +0.878 |
| medium_broadway | +0.916 |
| suited_aces | +0.799 |
| suited_connectors | +0.920 |
| suited_gappers | +0.934 |
| offsuit_connectors | +0.721 |
| weak_broadway | +0.688 |
| speculative_suited | +0.692 |
| trash | +0.908 |

- **trash-mass decodability R^2 = +0.908**  (mean over buckets +0.834)

## The encode-vs-decode gap (rows with a stated belief)
- mean **JS(stated belief, oracle) = 0.304 nats** (the decode-side miscalibration; higher = worse).
- oracle trash mass (truth): mean **0.682**
- model STATED trash mass: mean **0.049**  (|err| 0.634)
- PROBE-recovered trash from the residual: mean **0.682**  (|err| 0.009)

**Interpretation:** the residual probe recovers oracle trash mass MORE accurately than the model's own stated belief does. → the agent ENCODES more of the correct posterior than it STATES: miscalibration is (at least partly) a readout failure.

## Reading
- High oracle decodability (R^2) + high stated-belief JS = **knows-but-mis-states** (the citable Feng-et-al-style finding).
- The saved steering direction (residual→trash ridge weights) is used by `posterior_steering.py` to test whether ADDING it de-biases the stated belief / decision.
