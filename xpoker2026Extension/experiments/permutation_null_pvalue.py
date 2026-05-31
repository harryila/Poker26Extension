"""
Calibrated same-depth permutation null for a named head set (CPU; reads committed rows).

WHY (v2 weakness): the Qwen L19 necessity was p=3.8e-5 vs an EARLY-LAYER (L8) control but only
MARGINAL (p~0.06) vs ONE same-depth random 5-head draw. A reviewer reads that as "you localized a
LAYER, not a circuit." The fix is a proper permutation null: ablate MANY random k-head sets AT THE
SAME LAYER and ask where the named set's effect falls in that distribution -> a calibrated
empirical p-value and a z-score effect size.

This consumes a single inference_head_ablation cell dir that contains baseline_rows.jsonl, the named
condition (e.g. top5_rows.jsonl), and many random draws (r000_rows.jsonl ...). It reports, per
condition, the net any-flip over baseline AND the flip DESTINATION split (FOLD->CHECK vs FOLD->BET)
— the latter directly tests the C3 "bet-suppression not fold-circuit" reinterpretation: if the
named set's flips are not preferentially toward CHECK relative to random sets, it is not a
fold-specific circuit.

Usage:
    python -m experiments.permutation_null_pvalue \
        --cell results/inference_head_ablation/qwen8b_l19_permnull_clean_legal_fold \
        --named top5 --out results/inference_head_ablation/PERMNULL_qwen_l19.md
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os


def _verbs(path):
    """(seed,decision_idx) -> replay_verb."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[(r.get("seed"), r.get("decision_idx"))] = r.get("replay_verb")
    return out


def _flip_stats(base, cond):
    keys = set(base) & set(cond)
    fold_keys = [k for k in keys if base[k] == "FOLD"]
    n = len(fold_keys)
    if n == 0:
        return None
    flips = to_check = to_bet = 0
    for k in fold_keys:
        if cond[k] != "FOLD":
            flips += 1
            if cond[k] == "CHECK_OR_CALL":
                to_check += 1
            elif cond[k] == "BET_OR_RAISE":
                to_bet += 1
    return {"n_fold": n, "any_flip": flips / n,
            "to_check_frac": to_check / n, "to_bet_frac": to_bet / n,
            "to_check_of_flips": (to_check / flips) if flips else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--named", default="top5", help="condition name of the hypothesized head set")
    ap.add_argument("--rand-prefix", default="r", help="prefix of random-draw conditions (r000,...)")
    ap.add_argument("--min-draws", type=int, default=20,
                    help="minimum random draws for a calibrated null (default 20; the v3 run uses 50)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = _verbs(os.path.join(args.cell, "baseline_rows.jsonl"))
    if not base:
        raise SystemExit(f"no baseline_rows.jsonl in {args.cell}")
    named = _flip_stats(base, _verbs(os.path.join(args.cell, f"{args.named}_rows.jsonl")))
    if named is None:
        raise SystemExit(f"no {args.named}_rows.jsonl (or no FOLD targets) in {args.cell}")

    rand = []
    for f in sorted(globmod.glob(os.path.join(args.cell, f"{args.rand_prefix}*_rows.jsonl"))):
        nm = os.path.basename(f)[: -len("_rows.jsonl")]
        if nm in (args.named, "baseline", "control"):
            continue
        st = _flip_stats(base, _verbs(f))
        if st:
            rand.append((nm, st))
    k = len(rand)
    if k < args.min_draws:
        raise SystemExit(f"only {k} random draws found in {args.cell}; need >={args.min_draws} "
                         f"(lower with --min-draws for a smoke test)")

    import statistics as st
    rand_any = [s["any_flip"] for _, s in rand]
    rand_check = [s["to_check_frac"] for _, s in rand]
    mean_any, sd_any = st.mean(rand_any), (st.pstdev(rand_any) or 1e-9)
    # empirical one-sided p: how often does a random set flip >= the named set
    ge_any = sum(1 for v in rand_any if v >= named["any_flip"])
    p_any = (ge_any + 1) / (k + 1)
    z_any = (named["any_flip"] - mean_any) / sd_any
    # same for the FOLD->CHECK-specific axis (the "fold circuit" test)
    mean_chk, sd_chk = st.mean(rand_check), (st.pstdev(rand_check) or 1e-9)
    ge_chk = sum(1 for v in rand_check if v >= named["to_check_frac"])
    p_chk = (ge_chk + 1) / (k + 1)
    z_chk = (named["to_check_frac"] - mean_chk) / sd_chk

    md = ["# Same-depth permutation null — is the named head set special vs random heads at this layer?", ""]
    md.append(f"- Cell: `{args.cell}`  named set: **{args.named}**  random draws: **{k}**")
    md.append(f"- Baseline FOLD targets: {named['n_fold']}")
    md.append("")
    md.append("## Any-flip (replay verb != FOLD), net of the shared baseline")
    md.append(f"- named {args.named}: **{named['any_flip']*100:.1f}%**  vs random mean "
              f"{mean_any*100:.1f}% (sd {sd_any*100:.1f})")
    md.append(f"- **empirical p = {p_any:.3f}** ({ge_any}/{k} random sets flip >= named),  z = {z_any:+.2f}")
    md.append("")
    md.append("## FOLD->CHECK-specific (the 'fold circuit' test; C3 says it may be bet-suppression)")
    md.append(f"- named {args.named} FOLD->CHECK frac: **{named['to_check_frac']*100:.1f}%** "
              f"(of flips: {named['to_check_of_flips']*100:.0f}% go to CHECK, rest to BET)")
    md.append(f"- random mean FOLD->CHECK {mean_chk*100:.1f}% (sd {sd_chk*100:.1f}); "
              f"**empirical p = {p_chk:.3f}**, z = {z_chk:+.2f}")
    md.append("")
    md.append("## Reading")
    md.append("- **p_any < 0.05 (z>0)**: the named set ablates the verb MORE than same-depth random "
              "heads -> a genuine head-localized circuit, not just 'a deep layer matters'. "
              "**p_any ~ 0.5**: not special; necessity is layer-level/distributed, not a k-head circuit.")
    md.append("- **FOLD->CHECK test**: if the named set is NOT preferentially CHECK-directed vs random "
              "(p_chk high), the set is not a *fold* circuit — consistent with the C3 bet-suppression "
              "reinterpretation. Report any-flip AND destination split honestly.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
