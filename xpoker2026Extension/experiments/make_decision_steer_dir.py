"""
Build the CORRECT steering vector — the causal decision direction — for posterior_steering.py.

WHY (v2 post-mortem): the v2 steering used steer_trash_direction.npz = the residual->oracle-trash
RIDGE direction, which is ORTHOGONAL to the verb-decision axis (cos to centroid_diff = 0.006) and
was scaled by the full residual norm (158) so alpha 2/4/8 = 6-25x the natural check-fold gap ->
pure norm destruction. The causally-validated direction is centroid_diff = mean(X_check) -
mean(X_fold) (the same axis the 100%-flip patching moves along; cos(weight_vec, centroid_diff)=0.95).

This writes a steer npz whose `direction` is the UNIT decision axis and whose `resid_mean_norm` is
the RAW check-fold gap ||centroid_diff|| (~50 at Qwen L23). Then posterior_steering.py's
`vec * norm * alpha` makes alpha a multiple of ONE full check-minus-fold separation, so the sane
sweep is alpha in {0, 0.25, 0.5, 0.75, 1.0, 1.5} (NOT 2/4/8). No change to the tested steering code.

Input: either a probe `raw_residuals.npz` (keys X_cc, X_lf[, X_if]) OR a tagged recapture npz
(keys X, verb) — for layers (e.g. L19) that have no raw_residuals.npz.

Usage:
    # L23 from the existing probe residuals:
    python -m experiments.make_decision_steer_dir \
        --in results/direction_probe/qwen8b_l23/raw_residuals.npz --layer 23 \
        --out results/direction_probe/qwen8b_l23/steer_decision_direction.npz
    # L19 from a tagged recapture (run bet_matched_recapture --layer 19 first):
    python -m experiments.make_decision_steer_dir \
        --in results/direction_probe/qwen8b_l19/raw_residuals_tagged.npz --layer 19 \
        --out results/direction_probe/qwen8b_l19/steer_decision_direction.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _check_fold_means(npz):
    d = np.load(npz, allow_pickle=True)
    keys = set(d.files)
    if {"X_cc", "X_lf"} <= keys:
        check = d["X_cc"].astype(np.float64)
        fold = d["X_lf"].astype(np.float64)
        if "X_if" in keys and len(d["X_if"]):
            fold = np.vstack([fold, d["X_if"].astype(np.float64)])
        return check.mean(0), fold.mean(0), len(check), len(fold)
    if {"X", "verb"} <= keys:
        X = d["X"].astype(np.float64)
        verb = np.array([str(v).upper() for v in d["verb"]])
        is_check = np.array([("CALL" in v or "CHECK" in v) for v in verb])
        is_fold = verb == "FOLD"
        if is_check.sum() < 5 or is_fold.sum() < 5:
            raise SystemExit(f"too few check ({is_check.sum()}) or fold ({is_fold.sum()}) rows in {npz}")
        return X[is_check].mean(0), X[is_fold].mean(0), int(is_check.sum()), int(is_fold.sum())
    raise SystemExit(f"{npz} has neither (X_cc,X_lf) nor (X,verb); keys={sorted(keys)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mc, mf, ncheck, nfold = _check_fold_means(args.inp)
    centroid_diff = mc - mf                      # check - fold (sign: +dir pushes toward CHECK)
    gap = float(np.linalg.norm(centroid_diff))   # the natural decision displacement
    unit = centroid_diff / (gap + 1e-12)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             direction=unit.astype(np.float32),
             target="decision_check_minus_fold",
             layer=int(args.layer),
             resid_mean_norm=gap,            # alpha now scales in CHECK-FOLD GAP units
             n_check=ncheck, n_fold=nfold)
    print(f"[written] {args.out}")
    print(f"  check-fold gap ||centroid_diff|| = {gap:.2f}  (alpha units = multiples of this)")
    print(f"  n_check={ncheck} n_fold={nfold}")
    print(f"  => posterior_steering with this npz + --alphas 0 0.25 0.5 0.75 1.0 1.5 is sane")


if __name__ == "__main__":
    main()
