"""
Linear probe analysis: per-layer classification accuracy.

Loads cached hidden states and ground-truth labels, trains per-layer
linear probes, and generates accuracy-vs-layer curves.
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

try:
    from poker_env.interp.probing import ProbeDataset, LinearProbe
    PROBE_AVAILABLE = True
except ImportError:
    PROBE_AVAILABLE = False


def load_probe_data(path: str | Path) -> ProbeDataset:
    """
    Load probe dataset from a JSONL file.

    Expected format per line::

        {"layer_0": [0.1, 0.2, ...], "layer_1": [...], ..., "label": "premium_pairs"}
    """
    dataset = ProbeDataset()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get("label", "")
            layer_states = {}
            for key, value in record.items():
                if key.startswith("layer_"):
                    layer_idx = int(key.split("_")[1])
                    layer_states[layer_idx] = np.array(value, dtype=np.float32)
            if layer_states and label:
                dataset.add(layer_states, label)

    return dataset


def plot_accuracy_curve(
    results: dict[int, float],
    title: str = "Linear Probe Accuracy by Layer",
    output_path: str | None = None,
) -> None:
    """Plot per-layer probe accuracy."""
    if not MPL_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    layers = sorted(results.keys())
    accuracies = [results[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, accuracies, marker="o", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    peak_layer = layers[np.argmax(accuracies)]
    peak_acc = max(accuracies)
    ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.5,
               label=f"Peak: layer {peak_layer} ({peak_acc:.2%})")
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate linear probes")
    parser.add_argument("data", help="Probe dataset JSONL file")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default: 5)")
    parser.add_argument("--plot", type=str, default=None, help="Save accuracy plot")
    parser.add_argument("--json-out", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    if not PROBE_AVAILABLE:
        print("Error: scikit-learn required. pip install scikit-learn")
        return

    print(f"Loading probe data from {args.data}...")
    dataset = load_probe_data(args.data)
    print(f"  Samples: {dataset.num_samples}")
    print(f"  Layers:  {dataset.layers}")

    probe = LinearProbe()
    results = probe.evaluate(dataset, cv=args.cv)

    print(f"\nPer-layer accuracy ({args.cv}-fold CV):")
    for layer, acc in sorted(results.items()):
        bar = "#" * int(acc * 40)
        print(f"  Layer {layer:3d}: {acc:.3f}  {bar}")

    peak_layer = max(results, key=results.get)  # type: ignore
    print(f"\nPeak accuracy: layer {peak_layer} = {results[peak_layer]:.3f}")

    if args.plot:
        plot_accuracy_curve(results, output_path=args.plot)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"Saved results to {args.json_out}")


if __name__ == "__main__":
    main()
