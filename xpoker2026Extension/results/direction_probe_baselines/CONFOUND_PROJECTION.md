# Confound disentanglement — is the decision direction just 'facing a bet'?

CPU-only, committed data (`results/direction_probe/*/raw_residuals.npz`). Reproduce: `python -m experiments.confound_projection_analysis`.

**Question.** The verb label is near-collinear with `bet_to_call>0` (illegal_fold⟺bet=0, clean_legal_fold⟺bet>0), and a probe trained on `bet_to_call>0` matches the verb probe's accuracy. Is the learned decision direction `w` therefore just the 'facing-a-bet' axis?

**Test.** A pure bet-axis would place illegal_fold (bet=0) WITH check (bet=0). Instead we check whether `w` groups by VERB across bet regimes.

| Model/Layer | proj check/call | proj legal_fold (bet>0) | proj illegal_fold (bet=0) | both folds same side? | illegal_fold ⟂ check? | cos(w,verb-axis) | cos(w,bet-axis) | 1-D sep if/cc |
|---|---:|---:|---:|:--:|:--:|---:|---:|---:|
| Llama 8B L14 | +1.44 | -1.86 | -1.27 | YES | YES | **+0.982** | -0.322 | 100%/96% |
| Ministral 8B L16 | +1.14 | -2.07 | -1.50 | YES | YES | **+0.989** | -0.434 | 100%/100% |
| Qwen 8B L23 | +24.71 | -23.16 | -12.36 | YES | YES | **+0.948** | -0.396 | 100%/100% |

## Reading
- **Both fold types land on the same side, opposite to check** in all cells: `w` separates illegal_fold (bet=0) from check (also bet=0) — i.e. it groups by VERB, not by bet context. A pure bet-detector could not do this.
- **cos(w, verb-axis) ≈ +0.95–0.99 ≫ |cos(w, bet-axis)| ≈ 0.3–0.4**: the decision direction is geometrically aligned with the call−fold contrast, only weakly (and oppositely) with the bet contrast.
- **Conclusion:** the residual encodes BOTH verb and bet (both decodable ~0.99), but the learned decision direction is **verb-aligned, not a bet-context artifact**. This downgrades the confound from 'the direction is just bet' to 'the residual is information-rich; the decision axis is verb-specific.'
- **Caveat / why the GPU bet-matched probe still matters:** `w` was trained on data including these illegal_folds, so this is *necessary* (not fully sufficient) evidence. The clean test — a held-out, bet-matched probe (CALL@bet>0 vs FOLD@bet>0) and a bet-balanced probe — needs per-sample bet tagging via GPU residual re-capture (`experiments/bet_matched_probe.py`). Expected outcome given the geometry above: the direction survives bet-matching.
