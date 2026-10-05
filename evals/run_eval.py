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

from bernoulli.config import load_settings
from bernoulli.decide import decide
from bernoulli.scorer import HFScorer, Scorer
from bernoulli.types import DecideOptions, DecideRequest, State
from evals import metrics as metrics_mod
from evals.example import EvalExample

# All known dataset loaders. New datasets: add to this map + evals/datasets/<name>.py
LOADERS = {
    "sst2": "evals.datasets.sst2",
}


@dataclass
class EvalResult:
    dataset: str
    config: str
    model: str
    revision: str | None
    n_examples: int
    debias: str
    metrics: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


def _get_loader(name: str) -> object:
    if name not in LOADERS:
        raise SystemExit(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    return importlib.import_module(LOADERS[name])


def run(
    scorer: Scorer,
    examples: Iterable[EvalExample],
    *,
    debias: str = "reverse",
    dataset_name: str = "",
) -> EvalResult:
    """Score every example and return an EvalResult with metrics + latency."""
    gold: list[str] = []
    preds: list[dict[str, float]] = []
    latencies: list[int] = []

    for ex in examples:
        req = DecideRequest(
            state=State(text=ex.state_text),
            questions=[ex.question],
            options=DecideOptions(debias=debias, calibrated=False),  # type: ignore[arg-type]
        )
        t0 = time.perf_counter()
        resp = decide(req, scorer)
        latencies.append(int((time.perf_counter() - t0) * 1000))
        decision = resp.decisions[ex.question.id]
        if decision.type != "choice":
            raise RuntimeError(f"non-choice decision not supported in M3c (got {decision.type})")
        gold.append(ex.gold)
        preds.append(decision.distribution)

    return EvalResult(
        dataset=dataset_name,
        config=f"debias={debias}",
        model=scorer.model_id,
        revision=scorer.revision,
        n_examples=len(gold),
        debias=debias,
        metrics={
            "accuracy": metrics_mod.accuracy(gold, preds),
            "macro_f1": metrics_mod.macro_f1(gold, preds),
            "ece": metrics_mod.ece(gold, preds),
            "brier": metrics_mod.brier(gold, preds),
            "nll": metrics_mod.nll(gold, preds),
        },
        latency=metrics_mod.latency_summary(latencies),
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
    parser.add_argument("--debias", default="reverse", choices=["none", "reverse", "cyclic"])
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

    loader = _get_loader(args.dataset)
    examples = loader.load(limit=args.limit)  # type: ignore[attr-defined]
    result = run(scorer, examples, debias=args.debias, dataset_name=args.dataset)

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
