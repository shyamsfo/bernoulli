"""Fit temperature-scaling calibration from an eval sidecar.

Usage:
    python -m evals.fit_calibration --from evals/reports/sst2.json \
        --out calibration/qwen2.5-vl-7b.json

Reads the per-example predictions saved in the eval sidecar (gold, dist,
question_type), groups them by question type, fits one T per type with
LBFGS on NLL, and writes a Calibration JSON. If multiple sidecars are
passed, their predictions are concatenated before fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from bernoulli.calibrate import Calibration, fit_temperature, new_version, save_calibration


def _load_predictions_grouped_by_type(
    sidecar_paths: list[Path],
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, float]]], str, str | None]:
    """Returns (gold_by_type, preds_by_type, model, revision)."""
    gold_by_type: dict[str, list[str]] = defaultdict(list)
    preds_by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    model: str | None = None
    revision: str | None = None
    for path in sidecar_paths:
        data = json.loads(path.read_text())
        if model is None:
            model = data["model"]
            revision = data.get("revision")
        elif data["model"] != model:
            raise SystemExit(f"model mismatch: {path} has {data['model']!r}, expected {model!r}")
        for p in data["predictions"]:
            qt = p["question_type"]
            gold_by_type[qt].append(p["gold"])
            preds_by_type[qt].append(p["dist"])
    if model is None:
        raise SystemExit("no predictions found in sidecar(s)")
    return gold_by_type, preds_by_type, model, revision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fit_calibration")
    parser.add_argument(
        "--from",
        dest="sidecars",
        nargs="+",
        required=True,
        type=Path,
        help="one or more eval sidecar JSON files (with 'predictions')",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output path for the Calibration JSON",
    )
    args = parser.parse_args(argv)

    gold_by_type, preds_by_type, model, revision = _load_predictions_grouped_by_type(args.sidecars)

    temperatures: dict[str, float] = {}
    for qt in sorted(gold_by_type):
        t = fit_temperature(gold_by_type[qt], preds_by_type[qt])
        temperatures[qt] = t
        print(f"  {qt}: n={len(gold_by_type[qt])}, T={t:.4f}", file=sys.stderr)

    calibration = Calibration(
        model=model, revision=revision, version=new_version(), temperatures=temperatures
    )
    save_calibration(args.out, calibration)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
