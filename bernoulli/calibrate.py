"""Temperature-scaling calibrator.

Per-question-type scalar T, fit on NLL over a held-out set. Persisted to a
JSON file and loaded at decide() time when options.calibrated=True.

Vision doc §4: 'Temperature scaling, with one T per question type. Fit with
LBFGS on NLL over a held-out set.' We fit T >= 0.1 using scipy's bounded
minimize_scalar — simpler and equally effective for a 1-D convex problem.

Math note on fitting from probabilities. decide() returns softmax-of-logits
(no raw logits are stored). We fit by taking log(probs) as pseudo-logits
and dividing by T: `softmax(log(p) / T)`. Any constant offset in logits
cancels under softmax, so recovering T from stored probs is mathematically
identical to fitting on the original pre-softmax logits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

QuestionType = Literal["choice", "binary", "rating"]

_EPS = 1e-12
_T_MIN = 0.1
_T_MAX = 10.0

Distribution = dict[str, float]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _align(dists: Sequence[Distribution]) -> tuple[list[str], NDArray[np.float64]]:
    if not dists:
        return [], np.zeros((0, 0), dtype=np.float64)
    labels = list(dists[0].keys())
    label_set = set(labels)
    rows = np.zeros((len(dists), len(labels)), dtype=np.float64)
    for i, d in enumerate(dists):
        if set(d.keys()) != label_set:
            raise ValueError(
                f"example {i} has keys {sorted(d.keys())} vs expected {sorted(labels)}"
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


def _softmax_rows(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = mat - mat.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    result: NDArray[np.float64] = exp / exp.sum(axis=1, keepdims=True)
    return result


def _nll(pseudo_logits: NDArray[np.float64], onehot: NDArray[np.float64], t: float) -> float:
    probs = _softmax_rows(pseudo_logits / t)
    return float(-(onehot * np.log(np.clip(probs, _EPS, 1.0))).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_temperature(gold: Sequence[str], preds: Sequence[Distribution]) -> float:
    """Return the scalar T in [T_MIN, T_MAX] that minimizes NLL via scipy's bounded search.

    T > 1 cools an over-confident model; T < 1 sharpens an under-confident one.
    """
    from scipy.optimize import minimize_scalar

    if not gold:
        return 1.0
    labels, mat = _align(preds)
    pseudo_logits = np.log(np.clip(mat, _EPS, 1.0))
    onehot = _onehot(labels, gold)

    result = minimize_scalar(
        lambda t: _nll(pseudo_logits, onehot, t),
        bounds=(_T_MIN, _T_MAX),
        method="bounded",
    )
    return float(result.x)


def apply_temperature(probs: Distribution, t: float) -> Distribution:
    """T-scale a probability dict. Preserves key order."""
    labels = list(probs.keys())
    pseudo = np.log(np.clip([probs[k] for k in labels], _EPS, 1.0))
    scaled = pseudo / t
    scaled -= scaled.max()
    e = np.exp(scaled)
    normalized = e / e.sum()
    return {k: float(normalized[i]) for i, k in enumerate(labels)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """One temperature per question type for a specific model + revision."""

    model: str
    revision: str | None
    version: str  # ISO date, used in DecideResponse.calibration_version
    temperatures: dict[str, float] = field(default_factory=dict)

    def t_for(self, question_type: QuestionType) -> float:
        return self.temperatures.get(question_type, 1.0)


def new_version() -> str:
    return datetime.now(UTC).date().isoformat()


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2))


def load_calibration(path: Path) -> Calibration:
    data = json.loads(path.read_text())
    return Calibration(
        model=data["model"],
        revision=data.get("revision"),
        version=data["version"],
        temperatures=dict(data.get("temperatures", {})),
    )
