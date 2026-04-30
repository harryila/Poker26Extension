"""
Logit lens analysis: layer-by-layer belief formation.

Loads logit-lens sidecar data alongside enriched decisions and produces:
- Per-layer entropy curves (how uncertain is the model at each layer?)
- Crystallization layer (at which layer does the top prediction stabilize?)
- Heatmaps of layer vs. belief-relevant token predictions
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def load_sidecar(path: str | Path) -> list[dict]:
    """Load logit-lens JSONL sidecar."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_crystallization_layer(per_layer_top_tokens: list[list[str]]) -> int | None:
    """
    Find the earliest layer from which the top-1 token never changes.

    Args:
        per_layer_top_tokens: per_layer_top_tokens[layer][position] = token string

    Returns:
        layer index where prediction stabilizes, or None
    """
    if not per_layer_top_tokens:
        return None

    num_layers = len(per_layer_top_tokens)
    final_tokens = per_layer_top_tokens[-1]

    for start_layer in range(num_layers):
        stable = True
        for layer in range(start_layer, num_layers):
            if per_layer_top_tokens[layer] != final_tokens:
                stable = False
                break
        if stable:
            return start_layer

    return num_layers - 1


def analyze_logit_lens_data(records: list[dict]) -> dict:
    """Analyze a set of logit-lens records."""
    if not records:
        return {"num_records": 0}

    num_layers_list = []
    crystallization_layers = []
    mean_entropies_per_layer: dict[int, list[float]] = {}

    for rec in records:
        num_layers = rec.get("num_layers", 0)
        num_layers_list.append(num_layers)

        per_layer_entropy = rec.get("per_layer_entropy", [])
        for layer_idx, ent in enumerate(per_layer_entropy):
            if layer_idx not in mean_entropies_per_layer:
                mean_entropies_per_layer[layer_idx] = []
            mean_entropies_per_layer[layer_idx].append(ent)

        per_layer_top = rec.get("per_layer_top_tokens", [])
        cl = compute_crystallization_layer(per_layer_top)
        if cl is not None:
            crystallization_layers.append(cl)

    avg_entropy_curve = [
        float(np.mean(mean_entropies_per_layer[l]))
        for l in sorted(mean_entropies_per_layer.keys())
    ]

    return {
        "num_records": len(records),
        "num_layers": int(np.median(num_layers_list)) if num_layers_list else 0,
        "avg_entropy_per_layer": avg_entropy_curve,
        "crystallization_layer": {
            "mean": float(np.mean(crystallization_layers)) if crystallization_layers else None,
            "median": float(np.median(crystallization_layers)) if crystallization_layers else None,
            "std": float(np.std(crystallization_layers)) if crystallization_layers else None,
            "n": len(crystallization_layers),
        },
    }


def plot_entropy_curve(results: dict, output_path: str | None = None) -> None:
    """Plot average per-layer entropy."""
    if not MPL_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    curve = results.get("avg_entropy_per_layer", [])
    if not curve:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(curve)), curve, marker="o", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Entropy (nats)")
    ax.set_title("Logit Lens: Per-Layer Prediction Entropy")
    ax.grid(True, alpha=0.3)

    cl = results.get("crystallization_layer", {}).get("mean")
    if cl is not None:
        ax.axvline(cl, color="red", linestyle="--", alpha=0.7, label=f"Mean crystallization layer ({cl:.1f})")
        ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved entropy plot to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze logit lens sidecar data")
    parser.add_argument("sidecar", help="Logit-lens JSONL sidecar file")
    parser.add_argument("--plot", type=str, default=None, help="Save entropy plot to file")
    parser.add_argument("--json-out", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    records = load_sidecar(args.sidecar)
    results = analyze_logit_lens_data(records)

    print(f"Logit Lens Analysis ({results['num_records']} records)")
    print(f"  Layers: {results['num_layers']}")

    cl = results["crystallization_layer"]
    if cl["mean"] is not None:
        print(f"  Crystallization layer: {cl['mean']:.1f} +/- {cl['std']:.1f} (median={cl['median']:.1f}, n={cl['n']})")

    if args.plot:
        plot_entropy_curve(results, args.plot)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.json_out}")


if __name__ == "__main__":
    main()
