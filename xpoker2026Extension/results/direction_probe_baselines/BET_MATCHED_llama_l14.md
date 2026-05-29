# Bet-matched probe — is the decision direction real beyond 'facing a bet'?

- Tagged residuals: `results/direction_probe/llama8b_l14/raw_residuals_tagged.npz`  (N=757)
- bet>0: 453  bet=0: 304
- Probe: standardized L2-LogReg, 5-fold stratified CV. Floor = permuted labels.

| probe | n | CV acc | permuted floor | within-subset bet cross-task |
|---|---:|---:|---:|---:|
| A. facing a bet (bet>0): CALL vs legal_fold | 453 | 0.987 | 0.550 | n/a (bet constant) |
| B. no bet (bet=0): CHECK vs illegal_fold | 304 | 0.987 | 0.662 | n/a (bet constant) |
| (ref) all data: call/check vs fold | 757 | 0.991 | 0.504 | 1.000 |
| C. bet-balanced: call/check vs fold | 272 | 0.996 | 0.497 | — |

## Reading
- **If A and B CV acc ≫ permuted floor**, the verb is decodable with bet held constant → the decision representation is NOT just the facing-a-bet feature. (Expected, given CONFOUND_PROJECTION.md geometry.)
- **within-subset bet cross-task near chance** confirms bet was actually held constant within the regime (the matching worked).
- **bet-balanced probe (C) ≫ floor** is the single cleanest number to cite: it breaks the verb↔bet collinearity by construction.
