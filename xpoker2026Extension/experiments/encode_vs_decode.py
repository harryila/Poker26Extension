"""
Encode-vs-decode: does the agent ENCODE the Bayesian posterior but DECODE it
miscalibrated? (CPU analysis on a tagged recapture; the Tier-1 novel finding.)

Motivation (Feng/Russell/Steinhardt 2024, arXiv 2406.19501): LLMs can encode a faithful
world model yet *decode* it unfaithfully. The base paper shows poker agents' STATED beliefs
are severely miscalibrated (e.g. ~17% trash mass vs a ~66% oracle). The question this script
answers: is the correct StrategyAware posterior nonetheless LINEARLY DECODABLE from the
residual at L*? If yes, the miscalibration is a READOUT failure, not an absence of knowledge
— a sharper, more citable claim than "beliefs are wrong."

Inputs: `raw_residuals_tagged.npz` from experiments/bet_matched_recapture.py (now stores
`oracle_strategy_aware`, `oracle_card_only`, `agent_belief` as 14-vectors alongside X).

Outputs:
  * decodability of the oracle posterior from the residual (ridge, CV R^2; per-bucket;
    special focus on the 'trash' mass the model neglects);
  * the model's OWN stated miscalibration JS(agent_belief, oracle) for comparison;
  * the ENCODE-vs-DECODE gap: probe-recovered trash mass vs stated trash mass;
  * a saved STEERING direction (ridge weights toward oracle trash mass) for
    experiments/posterior_steering.py.

CPU/sklearn. Run after the GPU recapture:
    python -m experiments.encode_vs_decode \
        --tagged results/direction_probe/qwen8b_l23/raw_residuals_tagged.npz \
        --out results/direction_probe_baselines/ENCODE_VS_DECODE_qwen_l23.md \
        --save-direction results/direction_probe/qwen8b_l23/steer_trash_direction.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _js(p, q, eps=1e-9):
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)   # nats, in [0, ln2]


def _cv_r2_multioutput(X, Y, n_splits=5, seed=0, alpha=10.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        clf = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    # per-target R^2
    ss_res = ((Y - preds) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return r2, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tagged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-direction", default=None,
                    help="npz path to save the residual->oracle-trash ridge direction (steering vec)")
    ap.add_argument("--target", default="oracle_strategy_aware",
                    choices=["oracle_strategy_aware", "oracle_card_only"])
    args = ap.parse_args()

    d = np.load(args.tagged, allow_pickle=True)
    X = d["X"].astype(np.float64)
    order = [str(b) for b in d["bucket_order"]]
    oracle = d[args.target].astype(np.float64)
    belief = d["agent_belief"].astype(np.float64)
    trash_idx = order.index("trash") if "trash" in order else len(order) - 1

    # keep rows with a valid oracle
    ok = ~np.isnan(oracle).any(1)
    Xo, Yo = X[ok], oracle[ok]
    if len(Xo) < 30:
        raise SystemExit(f"too few rows with oracle ({len(Xo)}); recapture on a belief-elicited log")

    r2, preds = _cv_r2_multioutput(Xo, Yo)
    # decodable trash mass vs stated trash mass
    have_belief = ok & (~np.isnan(belief).any(1))
    js_stated = np.array([_js(belief[i], oracle[i]) for i in np.where(have_belief)[0]])
    stated_trash = belief[have_belief][:, trash_idx]
    oracle_trash = oracle[have_belief][:, trash_idx]
    # probe-recovered trash on those same rows (from the CV preds aligned to Xo/Yo)
    idx_in_ok = {orig: k for k, orig in enumerate(np.where(ok)[0])}
    rec_trash = np.array([preds[idx_in_ok[i]][trash_idx] for i in np.where(have_belief)[0]])

    md = ["# Encode-vs-decode: is the Bayesian posterior decodable from the residual?", ""]
    md.append(f"- Tagged residuals: `{args.tagged}`  (rows with oracle: {len(Xo)})")
    md.append(f"- Target posterior: `{args.target}`  (14 buckets; 'trash' index {trash_idx})")
    md.append("- Decodability = 5-fold CV R^2 of ridge regression residual → posterior probability.")
    md.append("")
    md.append("## Per-bucket decodability (CV R^2)")
    md.append("| bucket | CV R^2 |")
    md.append("|---|---:|")
    for b, r in zip(order, r2):
        md.append(f"| {b} | {r:+.3f} |")
    md.append(f"\n- **trash-mass decodability R^2 = {r2[trash_idx]:+.3f}**  "
              f"(mean over buckets {r2.mean():+.3f})")
    md.append("")
    if have_belief.sum() > 0:
        md.append("## The encode-vs-decode gap (rows with a stated belief)")
        md.append(f"- mean **JS(stated belief, oracle) = {js_stated.mean():.3f} nats** "
                  f"(the decode-side miscalibration; higher = worse).")
        md.append(f"- oracle trash mass (truth): mean **{oracle_trash.mean():.3f}**")
        md.append(f"- model STATED trash mass: mean **{stated_trash.mean():.3f}**  "
                  f"(|err| {np.abs(stated_trash-oracle_trash).mean():.3f})")
        md.append(f"- PROBE-recovered trash from the residual: mean **{rec_trash.mean():.3f}**  "
                  f"(|err| {np.abs(rec_trash-oracle_trash).mean():.3f})")
        probe_err = float(np.abs(rec_trash - oracle_trash).mean())
        stated_err = float(np.abs(stated_trash - oracle_trash).mean())
        better = probe_err < stated_err
        comp = "MORE" if better else "NOT more"
        concl = ("→ the agent ENCODES more of the correct posterior than it STATES: "
                 "miscalibration is (at least partly) a readout failure."
                 if better else "→ no encode-vs-decode gap on trash here.")
        md.append("")
        md.append(f"**Interpretation:** the residual probe recovers oracle trash mass "
                  f"{comp} accurately than the model's own stated belief does. {concl}")
    md.append("")
    md.append("## Reading")
    md.append("- High oracle decodability (R^2) + high stated-belief JS = **knows-but-mis-states** "
              "(the citable Feng-et-al-style finding).")
    md.append("- The saved steering direction (residual→trash ridge weights) is used by "
              "`posterior_steering.py` to test whether ADDING it de-biases the stated belief / decision.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")

    if args.save_direction:
        # Fit a single ridge on ALL oracle rows for the trash target -> direction in residual space.
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xo)
        clf = Ridge(alpha=10.0).fit(sc.transform(Xo), Yo[:, trash_idx])
        # map standardized coef back to raw residual space: w_raw = coef / scale
        w_raw = clf.coef_ / sc.scale_
        w_raw = w_raw / (np.linalg.norm(w_raw) + 1e-12)
        os.makedirs(os.path.dirname(args.save_direction), exist_ok=True)
        np.savez(args.save_direction,
                 direction=w_raw.astype(np.float32),
                 target="trash_mass",
                 layer=int(d["layer"]) if "layer" in d else -1,
                 resid_mean_norm=float(np.linalg.norm(Xo, axis=1).mean()))
        print(f"[written] steering direction -> {args.save_direction}")


if __name__ == "__main__":
    main()
