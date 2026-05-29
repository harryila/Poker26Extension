# Encode-vs-decode: is the Bayesian posterior decodable from the residual?

- Tagged residuals: `results/direction_probe/ministral8b_l16/raw_residuals_tagged.npz`  (rows with oracle: 514)
- Target posterior: `oracle_strategy_aware`  (14 buckets; 'trash' index 13)
- Decodability = 5-fold CV R^2 of ridge regression residual → posterior probability.

## Per-bucket decodability (CV R^2)
| bucket | CV R^2 |
|---|---:|
| premium_pairs | +0.931 |
| strong_pairs | +0.832 |
| medium_pairs | +0.630 |
| small_pairs | +0.729 |
| premium_broadway | +0.903 |
| strong_broadway | +0.918 |
| medium_broadway | +0.940 |
| suited_aces | +0.785 |
| suited_connectors | +0.956 |
| suited_gappers | +0.968 |
| offsuit_connectors | +0.572 |
| weak_broadway | +0.390 |
| speculative_suited | +0.742 |
| trash | +0.959 |

- **trash-mass decodability R^2 = +0.959**  (mean over buckets +0.804)

## The encode-vs-decode gap (rows with a stated belief)
- mean **JS(stated belief, oracle) = 0.304 nats** (the decode-side miscalibration; higher = worse).
- oracle trash mass (truth): mean **0.682**
- model STATED trash mass: mean **0.049**  (|err| 0.634)
- PROBE-recovered trash from the residual: mean **0.682**  (|err| 0.006)

**Interpretation:** the residual probe recovers oracle trash mass MORE accurately than the model's own stated belief does. → the agent ENCODES more of the correct posterior than it STATES: miscalibration is (at least partly) a readout failure.

## Reading
- High oracle decodability (R^2) + high stated-belief JS = **knows-but-mis-states** (the citable Feng-et-al-style finding).
- The saved steering direction (residual→trash ridge weights) is used by `posterior_steering.py` to test whether ADDING it de-biases the stated belief / decision.
