"""
Attention pattern analysis for poker LLM agents.

Loads attention snapshots and analyzes which input tokens (cards,
actions, instructions) the model attends to when generating decisions.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def load_attention_data(path: str | Path) -> list[dict]:
    """Load attention snapshot JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_category_fractions(records: list[dict]) -> dict[str, dict[str, float]]:
    """
    Aggregate attention category fractions across all records.

    Returns dict with mean and std for each category.
    """
    category_values: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        fracs = rec.get("category_fractions", {})
        for cat, frac in fracs.items():
            category_values[cat].append(frac)

    result = {}
    for cat, values in category_values.items():
        result[cat] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    return result


def plot_category_fractions(
    category_stats: dict[str, dict[str, float]],
    title: str = "Attention Distribution by Token Category",
    output_path: str | None = None,
) -> None:
    if not MPL_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    cats = sorted(category_stats.keys())
    means = [category_stats[c]["mean"] for c in cats]
    stds = [category_stats[c]["std"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_ylabel("Fraction of Total Attention")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("data", help="Attention snapshot JSONL file")
    parser.add_argument("--plot", type=str, default=None, help="Save bar chart")
    parser.add_argument("--json-out", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    records = load_attention_data(args.data)
    print(f"Loaded {len(records)} attention snapshots")

    stats = aggregate_category_fractions(records)

    print("\nAttention by token category:")
    for cat in sorted(stats.keys()):
        s = stats[cat]
        print(f"  {cat:20s}: {s['mean']:.3f} +/- {s['std']:.3f} (n={s['n']})")

    if args.plot:
        plot_category_fractions(stats, output_path=args.plot)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved results to {args.json_out}")


if __name__ == "__main__":
    main()
