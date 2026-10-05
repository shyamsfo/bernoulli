"""Eval driver: iterate a dataset, call decide(), compute metrics, write a report.

Usage:
    python -m evals.run_eval --dataset sst2 --limit 10         # smoke
    python -m evals.run_eval --dataset sst2 --debias reverse   # full
    python -m evals.run_eval --dataset sst2 --out evals/reports/sst2.md

Outputs: compact per-run markdown table and a JSON sidecar with raw metrics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from bernoulli.calibrate import Calibration, load_calibration
from bernoulli.config import load_settings
from bernoulli.decide import decide
from bernoulli.scorer import HFScorer, Scorer
from bernoulli.types import DecideOptions, DecideRequest, Decision, State
from evals import metrics as metrics_mod
from evals.baselines import generative_decide
from evals.example import EvalExample

Method = str  # "bernoulli" (logit path) | "generative" (text + parse baseline)

# All known dataset loaders. New datasets: add to this map + evals/datasets/<name>.py
LOADERS = {
    "sst2": "evals.datasets.sst2",
    "ag_news": "evals.datasets.ag_news",
    "boolq": "evals.datasets.boolq",
    "banking77": "evals.datasets.banking77",
}


@dataclass
class EvalResult:
    dataset: str
    config: str
    model: str
    revision: str | None
    n_examples: int
    method: str
    debias: str
    metrics: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    predictions: list[dict[str, object]] = field(default_factory=list)
    timestamp: str = ""


def _get_loader(name: str) -> object:
    if name not in LOADERS:
        raise SystemExit(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    return importlib.import_module(LOADERS[name])


def _to_distribution(decision: Decision) -> dict[str, float]:
    """Normalize any Decision kind to the {label: prob} dict metrics expect.

    - ChoiceDecision and RatingDecision already expose `.distribution`.
    - BinaryDecision only exposes a scalar `probability`; synthesize the pair.
    """
    if decision.type == "binary":
        p = decision.probability
        return {"Yes": p, "No": 1.0 - p}
    return decision.distribution


def run(
    scorer: Scorer,
    examples: Iterable[EvalExample],
    *,
    method: Method = "bernoulli",
    debias: str = "reverse",
    calibration: Calibration | None = None,
    dataset_name: str = "",
) -> EvalResult:
    """Score every example and return an EvalResult with metrics + latency.

    method='bernoulli' uses the logit path (bernoulli.decide); 'generative'
    uses the text-and-parse baseline (evals.baselines.generative_decide).
    """
    gold: list[str] = []
    preds: list[dict[str, float]] = []
    question_types: list[str] = []
    latencies: list[int] = []

    use_calibration = calibration is not None and method != "generative"
    for ex in examples:
        req = DecideRequest(
            state=State(text=ex.state_text),
            questions=[ex.question],
            options=DecideOptions(debias=debias, calibrated=use_calibration),  # type: ignore[arg-type]
        )
        t0 = time.perf_counter()
        if method == "generative":
            resp = generative_decide(req, scorer)  # type: ignore[arg-type]
        else:
            resp = decide(req, scorer, calibration=calibration)
        latencies.append(int((time.perf_counter() - t0) * 1000))
        decision = resp.decisions[ex.question.id]
        gold.append(ex.gold)
        preds.append(_to_distribution(decision))
        question_types.append(decision.type)

    if method == "generative":
        config = f"method={method}"
    else:
        parts = [f"method={method}", f"debias={debias}"]
        if calibration is not None:
            parts.append(f"calibrated={calibration.version}")
        config = "; ".join(parts)
    predictions: list[dict[str, object]] = [
        {"gold": g, "dist": d, "question_type": qt}
        for g, d, qt in zip(gold, preds, question_types, strict=True)
    ]
    return EvalResult(
        dataset=dataset_name,
        config=config,
        model=scorer.model_id,
        revision=scorer.revision,
        n_examples=len(gold),
        method=method,
        debias="none" if method == "generative" else debias,
        metrics={
            "accuracy": metrics_mod.accuracy(gold, preds),
            "macro_f1": metrics_mod.macro_f1(gold, preds),
            "ece": metrics_mod.ece(gold, preds),
            "brier": metrics_mod.brier(gold, preds),
            "nll": metrics_mod.nll(gold, preds),
        },
        latency=metrics_mod.latency_summary(latencies),
        predictions=predictions,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _render_report(result: EvalResult) -> str:
    lines = [
        f"# Eval report — {result.dataset}",
        "",
        f"- **Timestamp**: {result.timestamp}",
        f"- **Model**: `{result.model}` @ `{result.revision}`",
        f"- **Config**: {result.config}",
        f"- **Examples**: {result.n_examples}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| accuracy  | {result.metrics['accuracy']:.4f} |",
        f"| macro_f1  | {result.metrics['macro_f1']:.4f} |",
        f"| ece (15)  | {result.metrics['ece']:.4f} |",
        f"| brier     | {result.metrics['brier']:.4f} |",
        f"| nll       | {result.metrics['nll']:.4f} |",
        "",
        "## Latency",
        "",
        "| percentile | ms |",
        "|---|---|",
        f"| p50  | {result.latency['p50_ms']:.1f} |",
        f"| p95  | {result.latency['p95_ms']:.1f} |",
        f"| mean | {result.latency['mean_ms']:.1f} |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval")
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument(
        "--method",
        default="bernoulli",
        choices=["bernoulli", "generative"],
        help="scoring method: 'bernoulli' (logit path, default) or 'generative' (baseline)",
    )
    parser.add_argument("--debias", default="reverse", choices=["none", "reverse", "cyclic"])
    parser.add_argument(
        "--calibrate",
        type=Path,
        default=None,
        help="path to a Calibration JSON (fit via `python -m evals.fit_calibration`)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap number of examples")
    parser.add_argument("--out", type=Path, default=None, help="output markdown path")
    args = parser.parse_args(argv)

    settings = load_settings()
    scorer = HFScorer(
        settings.model_id,
        revision=settings.model_revision,
        dtype=settings.dtype,
        device=settings.device,
    )

    calibration = load_calibration(args.calibrate) if args.calibrate else None
    if calibration is not None and args.method == "generative":
        print(
            "warning: --calibrate is ignored for method=generative (baseline has no probabilities to scale)",
            file=sys.stderr,
        )

    loader = _get_loader(args.dataset)
    examples = loader.load(limit=args.limit)  # type: ignore[attr-defined]
    result = run(
        scorer,
        examples,
        method=args.method,
        debias=args.debias,
        calibration=calibration,
        dataset_name=args.dataset,
    )

    report = _render_report(result)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        sidecar = args.out.with_suffix(".json")
        sidecar.write_text(json.dumps(asdict(result), indent=2))
        print(f"\nwrote {args.out} and {sidecar}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
