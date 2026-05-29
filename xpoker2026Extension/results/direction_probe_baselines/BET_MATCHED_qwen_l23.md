# Bet-matched probe — is the decision direction real beyond 'facing a bet'?

- Tagged residuals: `results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz`  (N=685)
- bet>0: 424  bet=0: 261
- Probe: standardized L2-LogReg, 5-fold stratified CV. Floor = permuted labels.

| probe | n | CV acc | permuted floor | within-subset bet cross-task |
|---|---:|---:|---:|---:|
| A. facing a bet (bet>0): CALL vs legal_fold | 424 | 1.000 | 0.521 | n/a (bet constant) |
| B. no bet (bet=0): CHECK vs illegal_fold | 261 | 0.992 | 0.857 | n/a (bet constant) |
| (ref) all data: call/check vs fold | 685 | 0.999 | 0.519 | 0.999 |
| C. bet-balanced: call/check vs fold | 96 | 1.000 | 0.506 | — |

## Reading
- **If A and B CV acc ≫ permuted floor**, the verb is decodable with bet held constant → the decision representation is NOT just the facing-a-bet feature. (Expected, given CONFOUND_PROJECTION.md geometry.)
- **within-subset bet cross-task near chance** confirms bet was actually held constant within the regime (the matching worked).
- **bet-balanced probe (C) ≫ floor** is the single cleanest number to cite: it breaks the verb↔bet collinearity by construction.
