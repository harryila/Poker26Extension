"""
Linear probing on transformer hidden states.

Train lightweight linear classifiers at each layer to predict ground-truth
labels (opponent bucket, optimal action, hand strength). This reveals what
information the model has *internally* vs. what it *expresses* in output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ProbeDataset:
    """
    Collects (hidden_state, label) pairs during experiment runs.

    Call ``add()`` at each decision point with the layer-indexed hidden
    states and the ground-truth label.  After the run, call ``to_numpy()``
    to get arrays ready for sklearn.
    """

    # layer_idx -> list of hidden-state vectors (flattened to 1-D numpy)
    _hidden: dict[int, list[np.ndarray]] = field(default_factory=dict)
    _labels: list[str] = field(default_factory=list)

    def add(self, layer_hidden_states: dict[int, np.ndarray], label: str) -> None:
        """
        Record one data point.

        Args:
            layer_hidden_states: mapping layer_idx -> 1-D numpy vector
            label: ground-truth label string (e.g. bucket name or action)
        """
        if not layer_hidden_states:
            return

        if self._hidden:
            existing_keys = set(self._hidden.keys())
            new_keys = set(layer_hidden_states.keys())
            if existing_keys != new_keys:
                common = existing_keys & new_keys
                if not common:
                    import warnings
                    warnings.warn(
                        f"ProbeDataset.add: no common layers between existing "
                        f"{existing_keys} and new {new_keys}; sample dropped"
                    )
                    return
                import warnings
                missing = existing_keys - new_keys
                if missing:
                    warnings.warn(
                        f"ProbeDataset.add: dropping layers {missing} that have "
                        f"no data in this call; only common layers {common} kept"
                    )
                    self._hidden = {k: v for k, v in self._hidden.items() if k in common}
                layer_hidden_states = {k: v for k, v in layer_hidden_states.items() if k in common}

        for layer_idx, vec in layer_hidden_states.items():
            if layer_idx not in self._hidden:
                self._hidden[layer_idx] = []
            self._hidden[layer_idx].append(vec)
        self._labels.append(label)

    @property
    def num_samples(self) -> int:
        return len(self._labels)

    @property
    def layers(self) -> list[int]:
        return sorted(self._hidden.keys())

    def to_numpy(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays for a given layer."""
        X = np.stack(self._hidden[layer_idx])
        y = np.array(self._labels)
        return X, y


class LinearProbe:
    """
    Per-layer linear probe using sklearn LogisticRegression.

    Usage::

        probe = LinearProbe()
        results = probe.evaluate(dataset, cv=5)
        # results: dict[int, float]  mapping layer_idx -> mean CV accuracy
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LinearProbe: pip install scikit-learn")
        self.max_iter = max_iter
        self.C = C

    def evaluate(
        self,
        dataset: ProbeDataset,
        cv: int = 5,
    ) -> dict[int, float]:
        """
        Train + evaluate a probe at each layer via cross-validation.

        Returns:
            dict mapping layer_idx -> mean accuracy
        """
        results: dict[int, float] = {}
        le = LabelEncoder()

        for layer_idx in dataset.layers:
            X, y_str = dataset.to_numpy(layer_idx)
            y = le.fit_transform(y_str)

            if len(np.unique(y)) < 2:
                results[layer_idx] = 0.0
                continue

            clf = LogisticRegression(
                max_iter=self.max_iter,
                C=self.C,
                solver="lbfgs",
            )
            folds = min(cv, len(y))
            if folds < 2:
                clf.fit(X, y)
                results[layer_idx] = float(clf.score(X, y))
                continue
            scores = cross_val_score(clf, X, y, cv=folds, scoring="accuracy")
            results[layer_idx] = float(np.mean(scores))

        return results

    def train_and_predict(
        self,
        dataset: ProbeDataset,
        layer_idx: int,
    ) -> tuple[Any, np.ndarray]:
        """
        Train on full dataset and return (fitted model, predictions).
        Useful for confusion-matrix analysis.
        """
        X, y_str = dataset.to_numpy(layer_idx)
        le = LabelEncoder()
        y = le.fit_transform(y_str)

        if len(np.unique(y)) < 2:
            return None, None

        clf = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            solver="lbfgs",
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        return clf, le.inverse_transform(preds)
