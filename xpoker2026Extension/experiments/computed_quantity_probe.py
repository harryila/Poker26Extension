"""
Computed-quantity probe: does the residual encode a NON-trivial quantity the model must ESTIMATE,
beyond what the prompt trivially provides? (CPU; the redirect after the oracle-trash input-presence null.)

The encode-vs-decode "knows" claim died because the oracle posterior is a near-deterministic function
of prompt token counts (input-presence; oracle trash R^2 from 4 opponent-action counts ~0.96 > the
residual probe at every layer). The honest replacement targets a quantity that is NOT a low-order
function of the visible prompt: hand EQUITY given the (HIDDEN) true opponent cards
(equity = win + 0.5*tie from equity_given_true_hands). The model cannot read the opponent's cards, so
if the residual decodes true equity ABOVE the input-feature + early-layer baseline, the model is
genuinely ESTIMATING equity internally (a computed representation), not echoing the prompt.

Decisive test (partial-out): residualize BOTH the residual-probe prediction and the target on the
non-residual input features (opponent-action counts, board size, street, bet_to_call, hole ranks);
the surviving residual->target R^2 is the computed signal. Compare L2 (early) vs L* (late): a positive
late-minus-early Δ AFTER partialling = the model builds the estimate across depth.

Consumes two tagged recaptures (L2 and L*) produced by bet_matched_recapture.py (now saving `equity`
and `input_feats`). CPU/sklearn.

Usage:
    python -m experiments.computed_quantity_probe \
        --early results/direction_probe/qwen8b_l2/raw_residuals_tagged.npz \
        --late  results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz \
        --target equity \
        --out results/direction_probe_baselines/COMPUTED_QUANTITY_qwen.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _cv_r2(X, y, n_splits=5, seed=0, alpha=10.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot, pred


def _partial_out(M, F):
    """Return M with the linear contribution of features F regressed out (column-wise OLS resid)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    Fs = StandardScaler().fit_transform(F)
    if M.ndim == 1:
        M = M[:, None]
    res = M - LinearRegression().fit(Fs, M).predict(Fs)
    return res.squeeze()


def _load(npz, target):
    d = np.load(npz, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d[target].astype(np.float64) if target in d.files else None
    F = d["input_feats"].astype(np.float64) if "input_feats" in d.files else None
    layer = int(d["layer"]) if "layer" in d.files else -1
    return X, y, F, layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--early", required=True, help="L2 tagged recapture npz")
    ap.add_argument("--late", required=True, help="L* tagged recapture npz")
    ap.add_argument("--target", default="equity")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xe, ye, Fe, le = _load(args.early, args.target)
    Xl, yl, Fl, ll = _load(args.late, args.target)
    if ye is None or yl is None:
        raise SystemExit(f"target '{args.target}' not in the recapture npz — re-run bet_matched_recapture "
                         f"(it now saves equity + input_feats).")
    # keep rows with a finite target on each side independently
    me = np.isfinite(ye); ml = np.isfinite(yl)
    Xe, ye, Fe = Xe[me], ye[me], Fe[me]
    Xl, yl, Fl = Xl[ml], yl[ml], Fl[ml]

    # 1) raw decodability of the target from residual at each layer
    r2_e, _ = _cv_r2(Xe, ye)
    r2_l, _ = _cv_r2(Xl, yl)
    # 2) input-feature baseline (non-residual)
    r2_feat_e, _ = _cv_r2(Fe, ye)
    r2_feat_l, _ = _cv_r2(Fl, yl)
    # 3) DECISIVE: partial out input features from BOTH residual and target, re-probe
    Xe_p = _partial_out(Xe, Fe); ye_p = _partial_out(ye, Fe)
    Xl_p = _partial_out(Xl, Fl); yl_p = _partial_out(yl, Fl)
    r2_e_partial, _ = _cv_r2(Xe_p, ye_p)
    r2_l_partial, _ = _cv_r2(Xl_p, yl_p)

    md = [f"# Computed-quantity probe — does the residual ESTIMATE `{args.target}` beyond the prompt?", ""]
    md.append(f"- early: `{args.early}` (L{le}, n={len(ye)})   late: `{args.late}` (L{ll}, n={len(yl)})")
    md.append(f"- target = `{args.target}` (equity = win+0.5*tie given HIDDEN true opp cards — not in prompt)")
    md.append("")
    md.append("| probe | early L%d | late L%d | late−early |" % (le, ll))
    md.append("|---|---:|---:|---:|")
    md.append(f"| residual → {args.target} (raw CV R²) | {r2_e:+.3f} | {r2_l:+.3f} | {r2_l-r2_e:+.3f} |")
    md.append(f"| input-features → {args.target} (baseline) | {r2_feat_e:+.3f} | {r2_feat_l:+.3f} | — |")
    md.append(f"| **residual → {args.target}, input-feats PARTIALLED OUT** | **{r2_e_partial:+.3f}** | **{r2_l_partial:+.3f}** | **{r2_l_partial-r2_e_partial:+.3f}** |")
    md.append("")
    md.append("## Reading")
    md.append("- The **partialled** row is decisive: positive late R² there = the residual encodes "
              "target information NOT linearly available in the prompt features = a genuine internal "
              "ESTIMATE. **late−early > 0 (partialled)** = the model builds that estimate across depth "
              "(a 'computed representation' claim that survives the input-presence critique).")
    md.append("- If the partialled R² is ~0 or negative at the late layer, even equity is input-presence/"
              "noise here, and the 'computed' line should be dropped (report honestly).")
    md.append(f"- Context: raw residual decodability ({r2_l:+.3f}) vs input-feature baseline "
              f"({r2_feat_l:+.3f}); residual BEATS inputs only if {r2_l:.3f} > {r2_feat_l:.3f}.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
