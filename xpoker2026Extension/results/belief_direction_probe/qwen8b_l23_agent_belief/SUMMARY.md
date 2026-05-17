# Belief direction probe (B3) results

- Model: `Qwen/Qwen3-8B`
- Layer: **23**
- Belief source: `agent_belief`
- n_decisions: 300 (skipped due to missing belief: 0)

## Multi-output Ridge regression: residual → 14-d belief distribution
- in-sample overall R²: 0.985
- **held-out overall R² (mean over 5 folds): -2.007 ± 1.015** (this is the trustworthy generalization estimate)

| Bucket | in-sample R² | held-out R² (CV) |
|---|---:|---:|
| `premium_pairs` | +0.995 | -1.201 |
| `strong_pairs` | +0.985 | -1.512 |
| `medium_pairs` | +0.992 | +0.164 |
| `small_pairs` | +0.992 | +0.260 |
| `premium_broadway` | +0.996 | -0.769 |
| `strong_broadway` | +0.979 | -1.479 |
| `medium_broadway` | +0.988 | -0.501 |
| `weak_broadway` | +0.968 | -4.908 |
| `suited_connectors` | +0.977 | -1.880 |
| `suited_aces` | +0.995 | -1.304 |
| `suited_gappers` | +0.975 | -5.407 |
| `speculative_suited` | +0.986 | -1.581 |
| `offsuit_connectors` | +0.969 | -6.051 |
| `trash` | +0.995 | -1.932 |

## Principal directions of the belief subspace (SVD)

| Component | singular value | explained variance |
|---|---:|---:|
| 1 | 0.29 | 34.0% |
| 2 | 0.25 | 25.4% |
| 3 | 0.19 | 14.3% |
| 4 | 0.14 | 7.8% |
| 5 | 0.12 | 6.2% |

## Verb direction × belief subspace cosines

- **cos(w_verb, principal_belief_direction): -0.0041**
- cos(w_verb, top-5 belief PCs):
    1. -0.0041 (explained var: 34.0%)
    2. +0.0016 (explained var: 25.4%)
    3. +0.0065 (explained var: 14.3%)
    4. -0.0079 (explained var: 7.8%)
    5. -0.0010 (explained var: 6.2%)

- cos(w_verb, per-bucket belief direction):

| Bucket | cosine |
|---|---:|
| `premium_pairs` | -0.0067 |
| `strong_pairs` | +0.0009 |
| `medium_pairs` | +0.0064 |
| `small_pairs` | +0.0052 |
| `premium_broadway` | -0.0032 |
| `strong_broadway` | +0.0004 |
| `medium_broadway` | +0.0018 |
| `weak_broadway` | +0.0034 |
| `suited_connectors` | +0.0040 |
| `suited_aces` | +0.0004 |
| `suited_gappers` | +0.0073 |
| `speculative_suited` | +0.0008 |
| `offsuit_connectors` | -0.0013 |
| `trash` | -0.0033 |

## Reading guide

- **|cos(w_verb, principal_belief_direction)| < 0.2**: verb and belief are encoded ORTHOGONALLY at L*. The verb-decision direction is independent of the dominant belief direction. Implies belief and verb are separately represented and the L* circuit doesn't directly use belief content for the verb choice.
- **|cos(w_verb, principal_belief_direction)| > 0.5**: verb and belief share a substantial axis. Implies the belief representation is RECRUITED for the verb decision at L*.
- **Strong cosines on specific buckets**: the verb-direction is ALIGNED with specific belief buckets (e.g. `premium_pairs` positive, `trash` negative). Suggests the model uses certain hand-strength features more than others to decide the verb.
- **R² high (>0.4)**: belief is well-decoded from L* residual. L* carries belief information.
- **R² low (<0.1)**: belief is NOT linearly decodable from L*. Either belief lives at a different layer, or it's encoded non-linearly. (Note: belief inertia is a behavioral metric; this probe asks about the residual-stream representation.)
