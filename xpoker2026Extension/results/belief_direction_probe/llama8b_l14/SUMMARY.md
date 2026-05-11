# Belief direction probe (B3) results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Belief source: `oracle_strategy_aware`
- n_decisions: 300 (skipped due to missing belief: 0)

## Multi-output Ridge regression: residual → 14-d belief distribution
- overall R²: **0.756**
- per-bucket R²:

| Bucket | R² |
|---|---:|
| `premium_pairs` | +0.783 |
| `strong_pairs` | +0.693 |
| `medium_pairs` | +0.617 |
| `small_pairs` | +0.878 |
| `premium_broadway` | +0.799 |
| `strong_broadway` | +0.780 |
| `medium_broadway` | +0.772 |
| `weak_broadway` | +0.613 |
| `suited_connectors` | +0.822 |
| `suited_aces` | +0.754 |
| `suited_gappers` | +0.872 |
| `speculative_suited` | +0.690 |
| `offsuit_connectors` | +0.648 |
| `trash` | +0.870 |

## Principal directions of the belief subspace (SVD)

| Component | singular value | explained variance |
|---|---:|---:|
| 1 | 0.11 | 70.0% |
| 2 | 0.04 | 11.6% |
| 3 | 0.03 | 6.9% |
| 4 | 0.03 | 6.4% |
| 5 | 0.02 | 2.9% |

## Verb direction × belief subspace cosines

- **cos(w_verb, principal_belief_direction): +0.0164**
- cos(w_verb, top-5 belief PCs):
    1. +0.0164 (explained var: 70.0%)
    2. -0.0007 (explained var: 11.6%)
    3. -0.0050 (explained var: 6.9%)
    4. -0.0324 (explained var: 6.4%)
    5. +0.0033 (explained var: 2.9%)

- cos(w_verb, per-bucket belief direction):

| Bucket | cosine |
|---|---:|
| `premium_pairs` | -0.0172 |
| `strong_pairs` | -0.0052 |
| `medium_pairs` | -0.0096 |
| `small_pairs` | +0.0342 |
| `premium_broadway` | -0.0216 |
| `strong_broadway` | -0.0166 |
| `medium_broadway` | -0.0165 |
| `weak_broadway` | -0.0082 |
| `suited_connectors` | -0.0161 |
| `suited_aces` | -0.0186 |
| `suited_gappers` | -0.0184 |
| `speculative_suited` | -0.0012 |
| `offsuit_connectors` | -0.0129 |
| `trash` | +0.0154 |

## Reading guide

- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and belief are encoded ORTHOGONALLY at L*. The verb-decision direction is independent of the dominant belief direction. Implies belief and verb are separately represented and the L* circuit doesn't directly use belief content for the verb choice.
- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and belief share a substantial axis. Implies the belief representation is RECRUITED for the verb decision at L*.
- **Strong cosines on specific buckets**: the verb-direction is ALIGNED with specific belief buckets (e.g. `premium_pairs` positive, `trash` negative). Suggests the model uses certain hand-strength features more than others to decide the verb.
- **R² high (>0.4)**: belief is well-decoded from L* residual. L* carries belief information.
- **R² low (<0.1)**: belief is NOT linearly decodable from L*. Either belief lives at a different layer, or it's encoded non-linearly. (Note: belief inertia is a behavioral metric; this probe asks about the residual-stream representation.)
