"""
Bet-matched probe — the clean test of the audit crux (CPU analysis half).

THE CRUX (see CLAIMS_AND_IDENTIFICATION.md §3 and results/.../CONFOUND_PROJECTION.md):
the CHECK/CALL-vs-FOLD label is near-collinear with the observable game-state feature
`bet_to_call>0`. The CPU geometry analysis already shows the learned direction is
verb-aligned (cos +0.95–0.99) not bet-aligned, but the *clean* test holds bet CONSTANT:

  Regime A (facing a bet, bet_to_call>0):  CALL                vs  clean_legal_fold
  Regime B (no bet,        bet_to_call==0): CHECK               vs  illegal_fold

If a probe trained WITHIN a single bet regime still separates the verb far above the
permuted-label floor, the decision representation is real beyond "am I facing a bet?".
We also fit a bet-BALANCED probe (equal bet composition per verb class) and report the
within-regime `bet_to_call` cross-task accuracy (should fall toward chance once bet is
constant — a sanity check that the matching worked).

This is the CPU half: it consumes a TAGGED residual file produced on GPU by
`experiments/bet_matched_recapture.py` (one row per decision, with residual + bet_to_call
+ verb + bucket). Methodology mirrors the committed probes: standardize, L2-LogReg,
StratifiedKFold cross_val_score, plus a permuted-label floor and a random-direction baseline.

Run the self-test (no data / no model needed) to validate the logic:
    python -m experiments.bet_matched_probe --self-test

Run on real tagged residuals (after GPU recapture):
    python -m experiments.bet_matched_probe \
        --tagged results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz \
        --out results/direction_probe_baselines/BET_MATCHED_qwen_l23.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _cv_acc(X, y, n_splits=5, seed=0, C=1.0):
    """Standardized L2-logistic-regression CV accuracy (mirrors the committed probes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan"), 0
    n_splits = int(min(n_splits, np.bincount(y).min()))
    if n_splits < 2:
        return float("nan"), len(y)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=C))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    return float(sc.mean()), len(y)


def _permuted_floor(X, y, n_trials=20, seed=0):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_trials):
        yp = rng.permutation(y)
        a, _ = _cv_acc(X, yp, seed=int(rng.integers(1 << 30)))
        if not np.isnan(a):
            accs.append(a)
    return float(np.mean(accs)) if accs else float("nan")


def _bet_balanced_indices(verb, bet_pos, rng):
    """Subsample so each verb class has the same bet>0 : bet=0 ratio (break collinearity)."""
    verb = np.asarray(verb); bet_pos = np.asarray(bet_pos)
    idx = np.arange(len(verb))
    cells = {}
    for v in np.unique(verb):
        for b in (True, False):
            cells[(v, b)] = idx[(verb == v) & (bet_pos == b)]
    # target per (verb,bet) cell = min over cells that exist for both verbs
    n = min((len(c) for c in cells.values() if len(c) > 0), default=0)
    if n == 0:
        return idx  # cannot balance; return all
    keep = []
    for c in cells.values():
        if len(c):
            keep.append(rng.choice(c, size=min(n, len(c)), replace=False))
    return np.concatenate(keep) if keep else idx


def analyze(tagged_path: str, out_path: str):
    d = np.load(tagged_path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    verb = np.array([str(v).upper() for v in d["verb"]])
    bet = np.asarray(d["bet_to_call"], float)
    bet_pos = bet > 0
    # map verbs into the two binary tasks
    is_call = np.array([("CALL" in v or "CHECK" in v) for v in verb])
    is_fold = np.array([v == "FOLD" for v in verb])

    md = ["# Bet-matched probe — is the decision direction real beyond 'facing a bet'?", ""]
    md.append(f"- Tagged residuals: `{tagged_path}`  (N={len(X)})")
    md.append(f"- bet>0: {bet_pos.sum()}  bet=0: {(~bet_pos).sum()}")
    md.append("- Probe: standardized L2-LogReg, 5-fold stratified CV. Floor = permuted labels.")
    md.append("")

    def run(mask, label):
        sel = mask & (is_call | is_fold)
        Xs, ys = X[sel], is_call[sel].astype(int)
        acc, n = _cv_acc(Xs, ys)
        floor = _permuted_floor(Xs, ys) if n else float("nan")
        # within-subset bet cross-task (should be ~chance if bet is held constant)
        betacc = float("nan")
        if mask is not None and len(np.unique((bet_pos[sel]).astype(int))) > 1:
            betacc, _ = _cv_acc(Xs, bet_pos[sel].astype(int))
        return label, n, acc, floor, betacc

    rows = []
    rows.append(run(bet_pos, "A. facing a bet (bet>0): CALL vs legal_fold"))
    rows.append(run(~bet_pos, "B. no bet (bet=0): CHECK vs illegal_fold"))
    rows.append(run(np.ones(len(X), bool), "(ref) all data: call/check vs fold"))

    # bet-balanced
    rng = np.random.default_rng(0)
    bidx = _bet_balanced_indices(np.where(is_call, "CALL", "FOLD")[is_call | is_fold],
                                 bet_pos[is_call | is_fold], rng)
    base_idx = np.where(is_call | is_fold)[0]
    sel_idx = base_idx[bidx]
    Xb, yb = X[sel_idx], is_call[sel_idx].astype(int)
    acc_b, n_b = _cv_acc(Xb, yb)
    floor_b = _permuted_floor(Xb, yb) if n_b else float("nan")

    md.append("| probe | n | CV acc | permuted floor | within-subset bet cross-task |")
    md.append("|---|---:|---:|---:|---:|")
    for label, n, acc, floor, betacc in rows:
        md.append(f"| {label} | {n} | {acc:.3f} | {floor:.3f} | "
                  f"{('%.3f'%betacc) if not np.isnan(betacc) else 'n/a (bet constant)'} |")
    md.append(f"| C. bet-balanced: call/check vs fold | {n_b} | {acc_b:.3f} | {floor_b:.3f} | — |")
    md.append("")
    md.append("## Reading")
    md.append("- **If A and B CV acc ≫ permuted floor**, the verb is decodable with bet held "
              "constant → the decision representation is NOT just the facing-a-bet feature. "
              "(Expected, given CONFOUND_PROJECTION.md geometry.)")
    md.append("- **within-subset bet cross-task near chance** confirms bet was actually held "
              "constant within the regime (the matching worked).")
    md.append("- **bet-balanced probe (C) ≫ floor** is the single cleanest number to cite: it "
              "breaks the verb↔bet collinearity by construction.")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    if out_path:
        print(f"\n[written] {out_path}")


def self_test():
    """Validate the logic on synthetic data with KNOWN structure (no model/data needed)."""
    rng = np.random.default_rng(0)
    D = 64
    N = 400
    # ground-truth orthogonal directions
    v_verb = np.zeros(D); v_verb[0] = 1.0
    v_bet = np.zeros(D); v_bet[1] = 1.0
    verb = rng.integers(0, 2, N)          # 1=call/check, 0=fold
    bet = rng.integers(0, 2, N)           # 1=bet>0
    noise = rng.normal(0, 1.0, (N, D))

    def build(verb_coef, bet_coef):
        X = noise.copy()
        X += np.outer((verb * 2 - 1), v_verb) * verb_coef
        X += np.outer((bet * 2 - 1), v_bet) * bet_coef
        return X

    print("SELF-TEST 1: residual encodes BOTH verb and bet (the realistic case)")
    np.savez("/tmp/_bmtest1.npz", X=build(3.0, 3.0),
             verb=np.where(verb == 1, "CALL", "FOLD"),
             bet_to_call=bet.astype(float), bucket=np.array(["x"] * N))
    analyze("/tmp/_bmtest1.npz", "")
    print("\n   EXPECT: A & B (bet-matched) acc HIGH (verb still separable), bet cross-task ~chance.\n")

    print("SELF-TEST 2: residual encodes ONLY bet, verb is collinear with bet (confound case)")
    # make verb == bet so the only real signal is bet; verb has no own direction
    verb2 = bet.copy()
    X2 = noise + np.outer((bet * 2 - 1), v_bet) * 3.0
    np.savez("/tmp/_bmtest2.npz", X=X2,
             verb=np.where(verb2 == 1, "CALL", "FOLD"),
             bet_to_call=bet.astype(float), bucket=np.array(["x"] * N))
    analyze("/tmp/_bmtest2.npz", "")
    print("\n   EXPECT: A & B bet-matched acc ~chance (no verb signal once bet is fixed) -> "
          "this is what a TRUE confound looks like. Our real data should look like TEST 1.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tagged", help="tagged residual npz (X, verb, bet_to_call, bucket)")
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.tagged:
        analyze(args.tagged, args.out)
    else:
        ap.error("pass --self-test or --tagged")


if __name__ == "__main__":
    main()
