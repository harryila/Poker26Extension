# Belief direction probe (B3) results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Belief source: `oracle_strategy_aware`
- n_decisions: 300 (skipped due to missing belief: 0)

## Multi-output Ridge regression: residual → 14-d belief distribution
- overall R²: **0.550**
- per-bucket R²:

| Bucket | R² |
|---|---:|
| `premium_pairs` | +0.549 |
| `strong_pairs` | +0.371 |
| `medium_pairs` | +0.323 |
| `small_pairs` | +0.843 |
| `premium_broadway` | +0.565 |
| `strong_broadway` | +0.576 |
| `medium_broadway` | +0.583 |
| `weak_broadway` | +0.375 |
| `suited_connectors` | +0.670 |
| `suited_aces` | +0.474 |
| `suited_gappers` | +0.720 |
| `speculative_suited` | +0.535 |
| `offsuit_connectors` | +0.474 |
| `trash` | +0.643 |

## Principal directions of the belief subspace (SVD)

| Component | singular value | explained variance |
|---|---:|---:|
| 1 | 0.09 | 70.9% |
| 2 | 0.04 | 14.4% |
| 3 | 0.03 | 6.2% |
| 4 | 0.02 | 4.2% |
| 5 | 0.02 | 2.6% |

## Verb direction × belief subspace cosines

- **cos(w_verb, principal_belief_direction): +0.0466**
- cos(w_verb, top-5 belief PCs):
    1. +0.0466 (explained var: 70.9%)
    2. -0.0065 (explained var: 14.4%)
    3. -0.0561 (explained var: 6.2%)
    4. -0.1126 (explained var: 4.2%)
    5. +0.0841 (explained var: 2.6%)

- cos(w_verb, per-bucket belief direction):

| Bucket | cosine |
|---|---:|
| `premium_pairs` | -0.0618 |
| `strong_pairs` | -0.0236 |
| `medium_pairs` | -0.0188 |
| `small_pairs` | +0.1268 |
| `premium_broadway` | -0.0639 |
| `strong_broadway` | -0.0598 |
| `medium_broadway` | -0.0479 |
| `weak_broadway` | -0.0123 |
| `suited_connectors` | -0.0587 |
| `suited_aces` | -0.0691 |
| `suited_gappers` | -0.0618 |
| `speculative_suited` | -0.0339 |
| `offsuit_connectors` | -0.0200 |
| `trash` | +0.0451 |

## Reading guide

- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and belief are encoded ORTHOGONALLY at L*. The verb-decision direction is independent of the dominant belief direction. Implies belief and verb are separately represented and the L* circuit doesn't directly use belief content for the verb choice.
- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and belief share a substantial axis. Implies the belief representation is RECRUITED for the verb decision at L*.
- **Strong cosines on specific buckets**: the verb-direction is ALIGNED with specific belief buckets (e.g. `premium_pairs` positive, `trash` negative). Suggests the model uses certain hand-strength features more than others to decide the verb.
- **R² high (>0.4)**: belief is well-decoded from L* residual. L* carries belief information.
- **R² low (<0.1)**: belief is NOT linearly decodable from L*. Either belief lives at a different layer, or it's encoded non-linearly. (Note: belief inertia is a behavioral metric; this probe asks about the residual-stream representation.)
