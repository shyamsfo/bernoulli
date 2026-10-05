"""Classification + calibration metrics.

Pure numpy — no scikit-learn — so the eval harness runs with the base install.
All metrics operate on parallel lists of (gold_label, pred_distribution_dict)
pairs. The dict is {option_string: probability}, matching the DecideResponse
shape. The gold_label is one of the option_strings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Distribution = dict[str, float]


def _align(dists: Sequence[Distribution]) -> tuple[list[str], NDArray[np.float64]]:
    """Stack distributions into a matrix of shape (N, K) in a stable label order.

    The label order is taken from the first distribution (every example must
    expose the same keys, which is true within a single dataset).
    """
    if not dists:
        return [], np.zeros((0, 0), dtype=np.float64)
    labels = list(dists[0].keys())
    label_set = set(labels)
    rows = np.zeros((len(dists), len(labels)), dtype=np.float64)
    for i, d in enumerate(dists):
        if set(d.keys()) != label_set:
            raise ValueError(
                f"example {i} has keys {sorted(d.keys())} but expected {sorted(labels)}"
            )
        for j, label in enumerate(labels):
            rows[i, j] = d[label]
    return labels, rows


def _onehot(labels: list[str], gold: Sequence[str]) -> NDArray[np.float64]:
    index = {label: i for i, label in enumerate(labels)}
    onehot = np.zeros((len(gold), len(labels)), dtype=np.float64)
    for i, g in enumerate(gold):
        if g not in index:
            raise ValueError(f"gold label {g!r} at index {i} not in label set {labels}")
        onehot[i, index[g]] = 1.0
    return onehot


def accuracy(gold: Sequence[str], preds: Sequence[Distribution]) -> float:
    """Fraction of examples where argmax(pred) == gold."""
    if not gold:
        return 0.0
    labels, mat = _align(preds)
    predicted = [labels[int(i)] for i in np.argmax(mat, axis=1)]
    return float(np.mean([p == g for p, g in zip(predicted, gold, strict=True)]))


def macro_f1(gold: Sequence[str], preds: Sequence[Distribution]) -> float:
    """Macro-averaged F1: per-class F1 averaged with equal weight."""
    labels, mat = _align(preds)
    if not labels:
        return 0.0
    predicted = [labels[int(i)] for i in np.argmax(mat, axis=1)]
    f1s = []
    for label in labels:
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold, strict=True) if p != label and g == label)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp) / denom if denom else 0.0)
    return float(np.mean(f1s))


def ece(gold: Sequence[str], preds: Sequence[Distribution], n_bins: int = 15) -> float:
    """Expected calibration error with equal-width probability bins.

    Measures |accuracy - mean_confidence| within each confidence bin,
    weighted by bin size. Lower is better; 0 is perfect calibration.
    """
    if not gold:
        return 0.0
    labels, mat = _align(preds)
    pred_idx = np.argmax(mat, axis=1)
    confidences = mat[np.arange(len(mat)), pred_idx]
    correct = np.asarray(
        [labels[int(i)] == g for i, g in zip(pred_idx, gold, strict=True)],
        dtype=np.float64,
    )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # include the right edge in the last bin
        in_bin = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        total += (n / len(gold)) * abs(bin_acc - bin_conf)
    return float(total)


def brier(gold: Sequence[str], preds: Sequence[Distribution]) -> float:
    """Multi-class Brier score: mean squared error between pred and onehot."""
    if not gold:
        return 0.0
    labels, mat = _align(preds)
    onehot = _onehot(labels, gold)
    return float(np.mean(np.sum((mat - onehot) ** 2, axis=1)))


def nll(gold: Sequence[str], preds: Sequence[Distribution], eps: float = 1e-12) -> float:
    """Average negative log-likelihood of the gold label under the prediction."""
    if not gold:
        return 0.0
    labels, mat = _align(preds)
    index = {label: i for i, label in enumerate(labels)}
    gold_probs = np.asarray([mat[i, index[g]] for i, g in enumerate(gold)], dtype=np.float64)
    return float(-np.mean(np.log(np.clip(gold_probs, eps, 1.0))))


def latency_summary(latencies_ms: Sequence[int]) -> dict[str, float]:
    """p50 and p95 latency across examples."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }
