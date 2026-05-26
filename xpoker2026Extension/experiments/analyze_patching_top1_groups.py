"""
Summarize patched top-1 verb families from causal_patching ``by_pair.csv``.

For BET→illegal_FOLD (and similar verb-generality cells), the auto-SUMMARY
emphasizes Δ(CHECK−FOLD). This script reports ``patched_top1_group`` rates
(e.g. fraction → BET_RAISE) per layer and overall.

Usage::

    python -m experiments.analyze_patching_top1_groups \\
        --by-pair results/causal_patching/ministral8b_bet_to_illegal_fold_l16/by_pair.csv

    python -m experiments.analyze_patching_top1_groups \\
        --results-dir results/causal_patching --glob '*bet_to_illegal_fold*'
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from collections import Counter, defaultdict
from pathlib import Path


def _summarize_csv(path: Path) -> dict:
    by_layer: dict[int, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    n = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            grp = row.get("patched_top1_group") or "UNKNOWN"
            overall[grp] += 1
            layer = int(row["layer"])
            by_layer[layer][grp] += 1
    return {
        "path": str(path),
        "n_pairs": n,
        "overall": dict(overall),
        "overall_frac": {k: v / n for k, v in overall.items()} if n else {},
        "by_layer": {
            str(L): {
                "counts": dict(c),
                "frac": {k: v / sum(c.values()) for k, v in c.items()},
            }
            for L, c in sorted(by_layer.items())
        },
    }


def _write_summary_md(out_path: Path, summaries: list[dict], title: str) -> None:
    lines = [f"# {title}", ""]
    for s in summaries:
        lines.append(f"## `{Path(s['path']).name}`")
        lines.append("")
        lines.append(f"- n_pairs: {s['n_pairs']}")
        if not s["n_pairs"]:
            lines.append("")
            continue
        lines.append("")
        lines.append("| patched_top1_group | count | fraction |")
        lines.append("|---|---:|---:|")
        for grp, frac in sorted(
            s["overall_frac"].items(), key=lambda x: -x[1],
        ):
            cnt = s["overall"][grp]
            lines.append(f"| {grp} | {cnt} | {frac*100:.1f}% |")
        lines.append("")
        if s.get("by_layer"):
            lines.append("### By layer")
            lines.append("")
            for layer, data in s["by_layer"].items():
                top = max(data["frac"].items(), key=lambda x: x[1])
                lines.append(
                    f"- L={layer}: dominant `{top[0]}` "
                    f"({top[1]*100:.1f}% of {sum(data['counts'].values())} pairs)"
                )
            lines.append("")
        # Verb-generality readout for BET source cells
        bet_frac = s["overall_frac"].get("BET_RAISE", 0.0)
        check_frac = s["overall_frac"].get("CHECK_CALL", 0.0)
        fold_frac = s["overall_frac"].get("FOLD", 0.0)
        lines.append(
            f"**BET-generality readout**: top-1 → BET_RAISE "
            f"**{bet_frac*100:.1f}%** | CHECK_CALL {check_frac*100:.1f}% "
            f"| FOLD {fold_frac*100:.1f}%"
        )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Top-1 family summary from patching by_pair.csv"
    )
    parser.add_argument("--by-pair", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--glob", type=str, default="*bet_to_illegal_fold*")
    parser.add_argument("--out", type=str, default=None,
                        help="Write SUMMARY_top1_groups.md (default: alongside csv)")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.by_pair:
        paths.append(Path(args.by_pair))
    if args.results_dir:
        root = Path(args.results_dir)
        pattern = args.glob.strip("/")
        for p in sorted(root.rglob("by_pair.csv")):
            if fnmatch.fnmatch(p.parent.name, pattern):
                paths.append(p)

    if not paths:
        print("[abort] no by_pair.csv files found", flush=True)
        raise SystemExit(2)

    summaries = [_summarize_csv(p) for p in paths]
    for s in summaries:
        print(json.dumps(s, indent=2))

    if len(paths) == 1:
        out_md = Path(args.out) if args.out else paths[0].parent / "SUMMARY_top1_groups.md"
    else:
        out_md = Path(args.out) if args.out else Path(args.results_dir) / "SUMMARY_top1_groups.md"

    _write_summary_md(out_md, summaries, "Patched top-1 verb families")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
