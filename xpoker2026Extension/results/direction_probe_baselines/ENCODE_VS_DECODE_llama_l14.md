# Encode-vs-decode: is the Bayesian posterior decodable from the residual?

- Tagged residuals: `results/direction_probe/llama8b_l14/raw_residuals_tagged.npz`  (rows with oracle: 757)
- Target posterior: `oracle_strategy_aware`  (14 buckets; 'trash' index 13)
- Decodability = 5-fold CV R^2 of ridge regression residual → posterior probability.

## Per-bucket decodability (CV R^2)
| bucket | CV R^2 |
|---|---:|
| premium_pairs | +0.520 |
| strong_pairs | +0.327 |
| medium_pairs | +0.305 |
| small_pairs | +0.677 |
| premium_broadway | +0.509 |
| strong_broadway | +0.536 |
| medium_broadway | +0.507 |
| suited_aces | +0.414 |
| suited_connectors | +0.656 |
| suited_gappers | +0.657 |
| offsuit_connectors | +0.179 |
| weak_broadway | +0.006 |
| speculative_suited | +0.364 |
| trash | +0.627 |

- **trash-mass decodability R^2 = +0.627**  (mean over buckets +0.449)

## The encode-vs-decode gap (rows with a stated belief)
- mean **JS(stated belief, oracle) = 0.195 nats** (the decode-side miscalibration; higher = worse).
- oracle trash mass (truth): mean **0.659**
- model STATED trash mass: mean **0.659**  (|err| 0.322)
- PROBE-recovered trash from the residual: mean **0.658**  (|err| 0.024)

**Interpretation:** the residual probe recovers oracle trash mass MORE accurately than the model's own stated belief does. → the agent ENCODES more of the correct posterior than it STATES: miscalibration is (at least partly) a readout failure.

## Reading
- High oracle decodability (R^2) + high stated-belief JS = **knows-but-mis-states** (the citable Feng-et-al-style finding).
- The saved steering direction (residual→trash ridge weights) is used by `posterior_steering.py` to test whether ADDING it de-biases the stated belief / decision.
