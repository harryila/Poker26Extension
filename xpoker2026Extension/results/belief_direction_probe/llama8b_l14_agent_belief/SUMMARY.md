# Belief direction probe (B3) results

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Layer: **14**
- Belief source: `agent_belief`
- n_decisions: 300 (skipped due to missing belief: 0)

## Multi-output Ridge regression: residual → 14-d belief distribution
- in-sample overall R²: 0.641
- **held-out overall R² (mean over 5 folds): 0.137 ± 0.026** (this is the trustworthy generalization estimate)

| Bucket | in-sample R² | held-out R² (CV) |
|---|---:|---:|
| `premium_pairs` | +0.465 | +0.028 |
| `strong_pairs` | +0.613 | +0.178 |
| `medium_pairs` | +0.731 | +0.423 |
| `small_pairs` | +0.732 | +0.391 |
| `premium_broadway` | +0.593 | +0.015 |
| `strong_broadway` | +0.552 | -0.032 |
| `medium_broadway` | +0.695 | +0.223 |
| `weak_broadway` | +0.632 | -0.300 |
| `suited_connectors` | +0.677 | +0.137 |
| `suited_aces` | +0.556 | +0.075 |
| `suited_gappers` | +0.684 | +0.184 |
| `speculative_suited` | +0.711 | +0.109 |
| `offsuit_connectors` | +0.640 | +0.157 |
| `trash` | +0.692 | +0.326 |

## Principal directions of the belief subspace (SVD)

| Component | singular value | explained variance |
|---|---:|---:|
| 1 | 2.12 | 77.0% |
| 2 | 0.67 | 7.7% |
| 3 | 0.52 | 4.5% |
| 4 | 0.42 | 3.0% |
| 5 | 0.33 | 1.9% |

## Verb direction × belief subspace cosines

- **cos(w_verb, principal_belief_direction): -0.0283**
- cos(w_verb, top-5 belief PCs):
    1. -0.0283 (explained var: 77.0%)
    2. +0.0163 (explained var: 7.7%)
    3. +0.0081 (explained var: 4.5%)
    4. -0.0065 (explained var: 3.0%)
    5. -0.0210 (explained var: 1.9%)

- cos(w_verb, per-bucket belief direction):

| Bucket | cosine |
|---|---:|
| `premium_pairs` | -0.0054 |
| `strong_pairs` | +0.0074 |
| `medium_pairs` | +0.0236 |
| `small_pairs` | +0.0260 |
| `premium_broadway` | +0.0123 |
| `strong_broadway` | +0.0100 |
| `medium_broadway` | +0.0203 |
| `weak_broadway` | +0.0220 |
| `suited_connectors` | +0.0176 |
| `suited_aces` | +0.0018 |
| `suited_gappers` | +0.0191 |
| `speculative_suited` | +0.0268 |
| `offsuit_connectors` | +0.0210 |
| `trash` | -0.0289 |

## Reading guide

- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and belief are encoded ORTHOGONALLY at L*. The verb-decision direction is independent of the dominant belief direction. Implies belief and verb are separately represented and the L* circuit doesn't directly use belief content for the verb choice.
- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and belief share a substantial axis. Implies the belief representation is RECRUITED for the verb decision at L*.
- **Strong cosines on specific buckets**: the verb-direction is ALIGNED with specific belief buckets (e.g. `premium_pairs` positive, `trash` negative). Suggests the model uses certain hand-strength features more than others to decide the verb.
- **R² high (>0.4)**: belief is well-decoded from L* residual. L* carries belief information.
- **R² low (<0.1)**: belief is NOT linearly decodable from L*. Either belief lives at a different layer, or it's encoded non-linearly. (Note: belief inertia is a behavioral metric; this probe asks about the residual-stream representation.)
