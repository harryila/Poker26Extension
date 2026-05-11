# Belief direction probe (B3) results

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Belief source: `oracle_strategy_aware`
- n_decisions: 300 (skipped due to missing belief: 0)

## Multi-output Ridge regression: residual → 14-d belief distribution
- overall R²: **0.999**
- per-bucket R²:

| Bucket | R² |
|---|---:|
| `premium_pairs` | +0.999 |
| `strong_pairs` | +0.999 |
| `medium_pairs` | +0.997 |
| `small_pairs` | +0.999 |
| `premium_broadway` | +0.999 |
| `strong_broadway` | +0.999 |
| `medium_broadway` | +0.999 |
| `weak_broadway` | +0.999 |
| `suited_connectors` | +0.999 |
| `suited_aces` | +0.998 |
| `suited_gappers` | +1.000 |
| `speculative_suited` | +0.999 |
| `offsuit_connectors` | +0.998 |
| `trash` | +1.000 |

## Principal directions of the belief subspace (SVD)

| Component | singular value | explained variance |
|---|---:|---:|
| 1 | 0.03 | 63.7% |
| 2 | 0.01 | 15.0% |
| 3 | 0.01 | 8.0% |
| 4 | 0.01 | 5.2% |
| 5 | 0.01 | 4.4% |

## Verb direction × belief subspace cosines

- **cos(w_verb, principal_belief_direction): +0.0067**
- cos(w_verb, top-5 belief PCs):
    1. +0.0067 (explained var: 63.7%)
    2. +0.0024 (explained var: 15.0%)
    3. -0.0037 (explained var: 8.0%)
    4. -0.0046 (explained var: 5.2%)
    5. -0.0031 (explained var: 4.4%)

- cos(w_verb, per-bucket belief direction):

| Bucket | cosine |
|---|---:|
| `premium_pairs` | -0.0015 |
| `strong_pairs` | +0.0049 |
| `medium_pairs` | -0.0026 |
| `small_pairs` | +0.0065 |
| `premium_broadway` | +0.0014 |
| `strong_broadway` | +0.0026 |
| `medium_broadway` | +0.0030 |
| `weak_broadway` | -0.0050 |
| `suited_connectors` | -0.0109 |
| `suited_aces` | -0.0006 |
| `suited_gappers` | -0.0100 |
| `speculative_suited` | -0.0044 |
| `offsuit_connectors` | -0.0064 |
| `trash` | +0.0062 |

## Reading guide

- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and belief are encoded ORTHOGONALLY at L*. The verb-decision direction is independent of the dominant belief direction. Implies belief and verb are separately represented and the L* circuit doesn't directly use belief content for the verb choice.
- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and belief share a substantial axis. Implies the belief representation is RECRUITED for the verb decision at L*.
- **Strong cosines on specific buckets**: the verb-direction is ALIGNED with specific belief buckets (e.g. `premium_pairs` positive, `trash` negative). Suggests the model uses certain hand-strength features more than others to decide the verb.
- **R² high (>0.4)**: belief is well-decoded from L* residual. L* carries belief information.
- **R² low (<0.1)**: belief is NOT linearly decodable from L*. Either belief lives at a different layer, or it's encoded non-linearly. (Note: belief inertia is a behavioral metric; this probe asks about the residual-stream representation.)
