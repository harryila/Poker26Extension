"""
Direction-probe baselines (A3):
  (a) RANDOM-direction probe: project residuals onto a random unit vector,
      compute classification accuracy. Should be near chance.
  (b) PERMUTED-LABEL probe: train the actual probe with shuffled labels.
      Should be near chance (50%).
  (c) CROSS-TASK probe: train the probe on a DIFFERENT label (a non-verb
      task derived from the decision metadata, e.g., "is the bet_to_call > 0?"
      from the obs). Should NOT predict the verb well — confirms the
      verb-direction is verb-specific, not just any high-information direction.

Loads cached `raw_residuals.npz` files produced by `decision_direction_probe.py`
plus the original enriched logs (for cross-task labels). Outputs a comparison
SUMMARY.md showing learned-probe CV accuracy vs each baseline.

Pure analysis on cached residuals — no GPU required after the residuals
were captured during the original direction-probe run.

Usage::

    python -m experiments.direction_probe_baselines \\
        --probe-npz results/direction_probe/llama8b_l14/raw_residuals.npz \\
        --enriched-log logs/cot_llama8b_t0_s42_*.jsonl.gz \\
                       logs/cot_llama8b_t0_s123_*.jsonl.gz \\
                       logs/cot_llama8b_t0_s456_*.jsonl.gz \\
        --out-md results/direction_probe_baselines/llama8b_l14.md \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.causal_patching import (  # noqa: E402
    classify_decision, _iter_decisions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Direction-probe baselines (random + cross-task)."
    )
    parser.add_argument("--probe-npz", required=True,
                        help="Cached raw_residuals.npz from decision_direction_probe.")
    parser.add_argument("--enriched-log", required=True, nargs="+",
                        help="Enriched logs used by the probe (for cross-task "
                             "labels keyed by hand_id). Must be the same logs.")
    parser.add_argument("--n-random-trials", type=int, default=20,
                        help="Number of random-direction trials to average.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--l2-c", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--cross-task-feature",
        default="bet_to_call",
        choices=["bet_to_call", "position", "street_preflop", "pot_size",
                 "is_first_decision"],
        help=(
            "Which feature to use as the cross-task baseline label. "
            "`bet_to_call` (legacy default) is correlated with verb (FOLD "
            "decisions almost always have bet_to_call > 0), so cross-task "
            "accuracy will be high for spurious reasons. `position` "
            "(BB vs SB) is essentially a coin flip and uncorrelated with "
            "verb — preferred for the writeup. `street_preflop`, "
            "`pot_size` (binary above-median), `is_first_decision` are "
            "additional independent options."
        ),
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help=(
            "Upsample the minority class to match the majority before CV. "
            "Use for models with extreme verb-class imbalance (e.g. "
            "Ministral non-CoT where CHECK is ~10% of decisions) — without "
            "this, permuted-label and random-direction baselines achieve "
            "majority-class accuracy and become uninformative."
        ),
    )
    args = parser.parse_args()

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    npz = np.load(args.probe_npz)
    Xc = npz["X_cc"].astype(np.float32)
    Xf = npz["X_lf"].astype(np.float32)
    Xi = npz["X_if"].astype(np.float32) if "X_if" in npz.files else None
    print(f"[init] residuals: clean_CC={Xc.shape}, clean_LF={Xf.shape}, "
          f"illegal_FOLD={Xi.shape if Xi is not None else None}")

    X = np.concatenate([Xc, Xf], axis=0)
    y = np.concatenate([np.ones(len(Xc), dtype=int),
                        np.zeros(len(Xf), dtype=int)])
    hidden = X.shape[1]
    rng = np.random.default_rng(args.seed)

    # Class-imbalance upsampling, if requested. Mirrors the same indices for
    # the cross-task label later, since cross-task labels are derived from
    # the same per-record key list as the residual rows.
    upsample_indices = None
    if args.balance_classes:
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if min(n_pos, n_neg) < max(n_pos, n_neg):
            minority_class = 1 if n_pos < n_neg else 0
            majority_class = 1 - minority_class
            n_minority = (y == minority_class).sum()
            n_majority = (y == majority_class).sum()
            n_extra = int(n_majority - n_minority)
            minority_idx = np.where(y == minority_class)[0]
            picks = rng.integers(0, len(minority_idx), size=n_extra)
            extra_idx = minority_idx[picks]
            keep_idx = np.concatenate([np.arange(len(y)), extra_idx])
            upsample_indices = keep_idx
            X = X[keep_idx]
            y = y[keep_idx]
            print(
                f"[balance] upsampled minority class {minority_class}: "
                f"original (pos={n_pos}, neg={n_neg}) -> "
                f"balanced n={len(y)} ({(y == 1).sum()} pos, "
                f"{(y == 0).sum()} neg)"
            )
        else:
            print("[balance] classes already balanced, no upsampling")

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    def _cv(model, X_local, y_local):
        s = cross_val_score(model, X_local, y_local, cv=skf, scoring="accuracy")
        return float(s.mean()), float(s.std())

    # ---- (1) The actual learned probe -------------------------------------
    clf = LogisticRegression(C=args.l2_c, max_iter=2000, solver="lbfgs",
                             random_state=args.seed)
    learned_mean, learned_std = _cv(clf, X, y)
    print(f"[learned] CV accuracy: {learned_mean:.3f} ± {learned_std:.3f}")

    # ---- (2) Permuted-label probe ----------------------------------------
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    perm_mean, perm_std = _cv(clf, X, y_shuffled)
    print(f"[permuted-label] CV accuracy: {perm_mean:.3f} ± {perm_std:.3f}")

    # ---- (3) Random-direction projections + threshold classifier ---------
    # For each random unit vector, compute the projection threshold that
    # maximizes class separation, then evaluate accuracy.
    random_accs = []
    for trial in range(args.n_random_trials):
        w = rng.normal(size=hidden).astype(np.float32)
        w = w / (np.linalg.norm(w) + 1e-9)
        proj = X @ w
        # Best threshold: sort projections, try midpoints, pick the one with
        # max accuracy. (Cheap, exact for 1-D classification.)
        sorted_idx = np.argsort(proj)
        sorted_y = y[sorted_idx]
        sorted_proj = proj[sorted_idx]
        # Compute accuracies at every possible threshold.
        # For each split point i, predict 1 for proj > threshold, 0 otherwise.
        # We try both directions (positive and negative class assignment).
        n = len(y)
        best = max(0.5, np.mean(sorted_y == 1), np.mean(sorted_y == 0))
        cum1 = np.cumsum(sorted_y == 1)
        cum0 = np.cumsum(sorted_y == 0)
        total1 = cum1[-1]
        total0 = cum0[-1]
        for i in range(1, n):
            # If we say "everything below idx i is class 0, above is class 1":
            correct = cum0[i-1] + (total1 - cum1[i-1])
            acc = correct / n
            if acc > best:
                best = acc
            # Reverse direction:
            correct2 = cum1[i-1] + (total0 - cum0[i-1])
            acc2 = correct2 / n
            if acc2 > best:
                best = acc2
        random_accs.append(best)
    rand_mean = float(np.mean(random_accs))
    rand_std = float(np.std(random_accs))
    print(f"[random-direction] best-threshold accuracy "
          f"({args.n_random_trials} trials): {rand_mean:.3f} ± {rand_std:.3f}")

    # ---- (4) Cross-task probe: train on a non-verb label -----------------
    # Build hand-keyed dict of records to derive cross-task labels.
    rec_index = {}
    for log_path in args.enriched_log:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata") or {}
            if not am.get("raw_response"):
                continue
            key = f"{rec.get('hand_id', '?')}:{rec.get('decision_idx', '?')}"
            rec_index[key] = rec

    # The probe NPZ stored the residuals but NOT the per-record keys; we
    # have only n_clean_check_or_call and n_clean_legal_fold counts. Without
    # keys, we cannot cleanly join cross-task labels per-record. Workaround:
    # use the same "rng.sample()" order with the same seed to reproduce the
    # bucket sampling, then derive labels in that order.
    #
    # In practice, the probe driver iterates by_bucket["..."] in order from
    # the enriched logs and takes the first max_decisions_per_bucket records.
    # Mirror that here.
    cc_keys = []
    lf_keys = []
    n_cc_target = len(Xc)
    n_lf_target = len(Xf)
    for log_path in args.enriched_log:
        for rec in _iter_decisions(log_path):
            am = rec.get("action_metadata") or {}
            if not am.get("raw_response"):
                continue
            b = classify_decision(rec)
            if b == "clean_check_or_call" and len(cc_keys) < n_cc_target:
                cc_keys.append(f"{rec.get('hand_id', '?')}:{rec.get('decision_idx', '?')}")
            elif b == "clean_legal_fold" and len(lf_keys) < n_lf_target:
                lf_keys.append(f"{rec.get('hand_id', '?')}:{rec.get('decision_idx', '?')}")
            if len(cc_keys) >= n_cc_target and len(lf_keys) >= n_lf_target:
                break

    # Cross-task feature definitions. Each takes a record dict and returns
    # an int label. `bet_to_call` is the legacy choice but is highly
    # correlated with verb (all FOLD decisions have bet_to_call > 0); the
    # other choices are intended to be roughly independent of verb.
    def _bet_to_call(rec):
        obs = rec.get("obs") or {}
        return int(obs.get("bet_to_call", 0) > 0)

    def _position(rec):
        # Heads-up: BB vs SB. Roughly 50/50, uncorrelated with verb.
        obs = rec.get("obs") or {}
        pos = obs.get("position") or rec.get("position")
        if isinstance(pos, str):
            return int(pos.upper() in ("BB", "BIG_BLIND"))
        return int(bool(pos))

    def _street_preflop(rec):
        s = rec.get("street") or (rec.get("obs") or {}).get("street")
        if isinstance(s, str):
            return int(s.upper() == "PREFLOP")
        return 0

    def _pot_size(rec):
        # Binary above-median split computed from the bucket later; here
        # we just emit the raw pot. The wrapper computes the median.
        obs = rec.get("obs") or {}
        return float(obs.get("pot", 0))

    def _is_first_decision(rec):
        return int(int(rec.get("decision_idx", 0)) == 1)

    cross_task_feature_map = {
        "bet_to_call": ("bet_to_call > 0", _bet_to_call),
        "position": ("position == BB", _position),
        "street_preflop": ("street == PREFLOP", _street_preflop),
        "pot_size": ("pot > median", _pot_size),
        "is_first_decision": ("decision_idx == 1", _is_first_decision),
    }

    cross_task_summary = None
    if len(cc_keys) == n_cc_target and len(lf_keys) == n_lf_target:
        all_keys = cc_keys + lf_keys
        feat_label, feat_fn = cross_task_feature_map[args.cross_task_feature]
        try:
            raw_vals = np.array(
                [feat_fn(rec_index[k]) for k in all_keys], dtype=np.float64
            )
            if args.cross_task_feature == "pot_size":
                med = float(np.median(raw_vals))
                y_cross_orig = (raw_vals > med).astype(int)
            else:
                y_cross_orig = raw_vals.astype(int)

            # If we upsampled the residual matrix above, mirror the same
            # row indices into the cross-task label vector so they align.
            if upsample_indices is not None:
                y_cross = y_cross_orig[upsample_indices]
            else:
                y_cross = y_cross_orig

            n_pos_cross = int((y_cross == 1).sum())
            n_neg_cross = int((y_cross == 0).sum())
            if len(set(y_cross.tolist())) < 2:
                print(
                    f"[cross-task] {feat_label}: degenerate single-class "
                    f"(n_pos={n_pos_cross}, n_neg={n_neg_cross}) — skipping"
                )
            else:
                cross_mean, cross_std = _cv(clf, X, y_cross)
                cross_task_summary = {
                    "task": feat_label,
                    "n_pos": n_pos_cross,
                    "n_neg": n_neg_cross,
                    "cv_acc_mean": cross_mean,
                    "cv_acc_std": cross_std,
                }
                print(
                    f"[cross-task] {feat_label}: n_pos={n_pos_cross} "
                    f"n_neg={n_neg_cross} CV={cross_mean:.3f} ± {cross_std:.3f}"
                )
        except Exception as e:
            print(f"[cross-task] failed ({feat_label}): {e}")
    else:
        print(f"[cross-task] sample order mismatch — skipping (cc {len(cc_keys)}/{n_cc_target}, lf {len(lf_keys)}/{n_lf_target})")

    # ---- Save -----------------------------------------------------------
    summary = {
        "probe_npz": args.probe_npz,
        "n_samples": int(len(X)),
        "hidden_dim": int(hidden),
        "balanced_classes": bool(args.balance_classes),
        "cross_task_feature": args.cross_task_feature,
        "learned_probe_cv_acc": {"mean": learned_mean, "std": learned_std},
        "permuted_label_cv_acc": {"mean": perm_mean, "std": perm_std},
        "random_direction_best_threshold_acc": {
            "mean": rand_mean, "std": rand_std,
            "n_trials": args.n_random_trials,
        },
        "cross_task": cross_task_summary,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# Direction-probe baselines", ""]
    md.append(f"- Probe: `{args.probe_npz}`")
    md.append(f"- Samples: {len(X)}, hidden_dim: {hidden}")
    md.append(f"- Class balancing: `{'upsampled' if args.balance_classes else 'as-is (no balancing)'}`")
    md.append(f"- Cross-task feature: `{args.cross_task_feature}`")
    md.append("")
    md.append("## Probe accuracy comparison")
    md.append("")
    md.append("| Probe | CV accuracy | Notes |")
    md.append("|---|---:|---|")
    md.append(f"| **Learned probe** (verb labels) | "
              f"**{learned_mean:.3f} ± {learned_std:.3f}** | the actual probe |")
    md.append(f"| Permuted-label control | {perm_mean:.3f} ± {perm_std:.3f} | "
              "shuffled labels — chance (~0.50) expected |")
    md.append(f"| Random-direction (best threshold, {args.n_random_trials} trials) | "
              f"{rand_mean:.3f} ± {rand_std:.3f} | "
              "1-D classification on a random axis |")
    if cross_task_summary is not None:
        md.append(f"| Cross-task (`{cross_task_summary['task']}`) | "
                  f"{cross_task_summary['cv_acc_mean']:.3f} ± "
                  f"{cross_task_summary['cv_acc_std']:.3f} | "
                  f"different label, same residuals "
                  f"(n_pos={cross_task_summary['n_pos']}, n_neg={cross_task_summary['n_neg']}) |")
    md.append("")
    md.append("## Reading guide")
    md.append("")
    md.append("- **Learned ≫ permuted-label**: confirms the probe is learning "
              "from the residual, not memorizing the labels.")
    md.append("- **Learned ≫ random-direction**: confirms a *specific* direction "
              "encodes the verb decision; not just any 1-D projection works.")
    md.append("- **Cross-task accuracy notable**: the residuals also encode "
              "other situational features. If cross-task accuracy is also high, "
              "it doesn't *contradict* the verb-direction finding — residuals "
              "encode many things — but it shows the residuals are "
              "information-rich at this layer.")
    md.append("- **Cross-task accuracy near chance**: confirms the verb "
              "direction is task-specific, not generic high-info.")
    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
