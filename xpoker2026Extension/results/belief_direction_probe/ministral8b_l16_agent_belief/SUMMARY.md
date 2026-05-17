# Belief direction probe (B3) results

- Model: `mistralai/Ministral-8B-Instruct-2410`
- Layer: **16**
- Belief source: `agent_belief`
- n_decisions: 300 (skipped due to missing belief: 2)

## Multi-output Ridge regression: residual → 14-d belief distribution
- in-sample overall R²: 0.331
- **held-out overall R² (mean over 5 folds): -0.180 ± 0.540** (this is the trustworthy generalization estimate)

| Bucket | in-sample R² | held-out R² (CV) |
|---|---:|---:|
| `premium_pairs` | +0.410 | +0.180 |
| `strong_pairs` | +0.429 | +0.193 |
| `medium_pairs` | +0.400 | +0.208 |
| `small_pairs` | +0.353 | +0.113 |
| `premium_broadway` | +0.397 | +0.181 |
| `strong_broadway` | +0.311 | +0.060 |
| `medium_broadway` | +0.269 | -0.079 |
| `weak_broadway` | +0.225 | +0.063 |
| `suited_connectors` | +0.429 | +0.161 |
| `suited_aces` | +0.438 | +0.169 |
| `suited_gappers` | +0.412 | +0.158 |
| `speculative_suited` | +0.135 | -2.000 |
| `offsuit_connectors` | +0.296 | +0.076 |
| `trash` | +0.135 | -2.000 |

## Principal directions of the belief subspace (SVD)

| Component | singular value | explained variance |
|---|---:|---:|
| 1 | 0.25 | 75.8% |
| 2 | 0.10 | 11.0% |
| 3 | 0.06 | 4.5% |
| 4 | 0.05 | 3.3% |
| 5 | 0.03 | 1.5% |

## Verb direction × belief subspace cosines

- **cos(w_verb, principal_belief_direction): +0.0446**
- cos(w_verb, top-5 belief PCs):
    1. +0.0446 (explained var: 75.8%)
    2. +0.0008 (explained var: 11.0%)
    3. +0.0224 (explained var: 4.5%)
    4. -0.0125 (explained var: 3.3%)
    5. +0.0239 (explained var: 1.5%)

- cos(w_verb, per-bucket belief direction):

| Bucket | cosine |
|---|---:|
| `premium_pairs` | -0.0357 |
| `strong_pairs` | -0.0423 |
| `medium_pairs` | -0.0467 |
| `small_pairs` | +0.0423 |
| `premium_broadway` | -0.0361 |
| `strong_broadway` | -0.0306 |
| `medium_broadway` | -0.0245 |
| `weak_broadway` | -0.0405 |
| `suited_connectors` | +0.0377 |
| `suited_aces` | +0.0467 |
| `suited_gappers` | +0.0509 |
| `speculative_suited` | -0.0093 |
| `offsuit_connectors` | +0.0316 |
| `trash` | -0.0093 |

## Reading guide

- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and belief are encoded ORTHOGONALLY at L*. The verb-decision direction is independent of the dominant belief direction. Implies belief and verb are separately represented and the L* circuit doesn't directly use belief content for the verb choice.
- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and belief share a substantial axis. Implies the belief representation is RECRUITED for the verb decision at L*.
- **Strong cosines on specific buckets**: the verb-direction is ALIGNED with specific belief buckets (e.g. `premium_pairs` positive, `trash` negative). Suggests the model uses certain hand-strength features more than others to decide the verb.
- **R² high (>0.4)**: belief is well-decoded from L* residual. L* carries belief information.
- **R² low (<0.1)**: belief is NOT linearly decodable from L*. Either belief lives at a different layer, or it's encoded non-linearly. (Note: belief inertia is a behavioral metric; this probe asks about the residual-stream representation.)
