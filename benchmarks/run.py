"""Benchmark runner — one invocation = one benchmark x one or more baselines.

Usage:

    uv run python -m benchmarks.run academic/sst2 \
        --baselines bernoulli,deberta,bge-m3-lr \
        --limit 100 \
        --bernoulli-url http://127.0.0.1:8000

    # To include the Generative baseline, pass --with-generative (loads HFScorer):
    uv run python -m benchmarks.run academic/sst2 \
        --baselines bernoulli,generative

Output: writes `<benchmark>/results/<YYYY-MM-DD>.md` with metric tables,
per-baseline predictions sidecar, stability pair, and reproducibility
footer (model id, Bernoulli commit, dataset identifier, seed, hardware).

The leaderboard-level summary table in `benchmarks/README.md` is updated
by a separate task (M8 task 13). This runner only writes the
per-benchmark report.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from benchmarks.common.baselines import (
    Baseline,
    BernoulliHTTP,
    BGEm3LR,
    DeBERTaZeroshot,
    Distribution,
    Generative,
)
from benchmarks.common.dataset import BenchmarkExample
from benchmarks.common.metrics import (
    DEFAULT_COVERAGE_LEVELS,
    accuracy,
    brier,
    coverage_curve,
    ece,
    latency_summary,
    macro_f1,
    nll,
)
from benchmarks.common.stability import stability_run

_KNOWN_BASELINES = ("bernoulli", "generative", "deberta", "bge-m3-lr")


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass
class BaselineMetrics:
    """One row in the per-benchmark report."""

    name: str
    accuracy: float
    macro_f1: float
    ece_10: float
    ece_15: float
    brier: float
    nll: float
    coverage: dict[float, float]
    latency_ms: dict[str, float]
    stability_reorder: float | None
    stability_reword: float
    n_examples: int
    note: str = ""  # free text, e.g. "trained on 5k examples"


# ---------------------------------------------------------------------------
# Baseline factory
# ---------------------------------------------------------------------------


def _build_baseline(
    name: str,
    *,
    bernoulli_url: str,
    device: str,
) -> Baseline:
    """Instantiate a baseline by its short name. Factory kept dumb on purpose.

    `generative` is deliberately not handled here — it needs an HFScorer
    which the caller loads once and threads through. See _build_generative.
    """
    if name == "bernoulli":
        return BernoulliHTTP(base_url=bernoulli_url)
    if name == "deberta":
        return DeBERTaZeroshot.load_default(device=0 if device == "cuda" else -1)
    if name == "bge-m3-lr":
        return BGEm3LR.load_default(device=device)
    if name == "generative":
        raise ValueError("Generative baseline is instantiated separately; use --with-generative.")
    raise ValueError(f"Unknown baseline {name!r}; must be one of {_KNOWN_BASELINES}")


def _build_generative() -> Generative:
    """Load the HFScorer via the project's load_scorer factory and wrap it."""
    from bernoulli.config import load_settings
    from bernoulli.scorer import load_scorer

    scorer = load_scorer(load_settings())
    return Generative(scorer)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _resolve_benchmark_module(spec: str) -> Any:
    """Import a loader like 'academic/sst2' → benchmarks.academic.sst2."""
    return importlib.import_module(f"benchmarks.{spec.replace('/', '.')}")


def _run_one_baseline(
    name: str,
    baseline: Baseline,
    examples: list[BenchmarkExample],
    *,
    reword_stems: tuple[str, ...],
    train_examples: list[BenchmarkExample] | None,
    train_limit: int | None,
) -> BaselineMetrics:
    """Fit (if trainable) + predict + metrics + stability for one baseline."""
    note = ""
    if hasattr(baseline, "fit"):
        if not train_examples:
            raise RuntimeError(
                f"Baseline {name!r} needs a training set but load_train() returned no "
                f"examples. Skip this baseline for this benchmark."
            )
        used = train_examples[:train_limit] if train_limit else train_examples
        baseline.fit(used)  # type: ignore[attr-defined]
        note = f"trained on {len(used)} examples"

    import time

    preds: list[Distribution] = []
    latencies: list[int] = []
    for ex in examples:
        t0 = time.perf_counter()
        preds.append(baseline.predict(ex))
        latencies.append(int((time.perf_counter() - t0) * 1000))

    gold = [ex.gold for ex in examples]
    stability = stability_run(baseline, examples, reword_stems)

    return BaselineMetrics(
        name=name,
        accuracy=accuracy(gold, preds),
        macro_f1=macro_f1(gold, preds),
        ece_10=ece(gold, preds, n_bins=10),
        ece_15=ece(gold, preds, n_bins=15),
        brier=brier(gold, preds),
        nll=nll(gold, preds),
        coverage=coverage_curve(gold, preds),
        latency_ms=latency_summary(latencies),
        stability_reorder=stability.reorder_score,
        stability_reword=stability.reword_score,
        n_examples=len(examples),
        note=note,
    )


# ---------------------------------------------------------------------------
# Reproducibility footer
# ---------------------------------------------------------------------------


def _bernoulli_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _repro_footer(benchmark_spec: str, module: Any) -> dict[str, str]:
    return {
        "benchmark": benchmark_spec,
        "bernoulli_commit": _bernoulli_commit(),
        "model_id": os.environ.get("BERNOULLI_MODEL_ID", "unset"),
        "model_revision": os.environ.get("BERNOULLI_MODEL_REVISION", "unset"),
        "hardware": platform.node(),
        "python": platform.python_version(),
        "source": getattr(module, "SOURCE", benchmark_spec),
    }


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def write_report(
    out_path: Path,
    benchmark_spec: str,
    results: list[BaselineMetrics],
    repro: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append(f"# `{benchmark_spec}` — results\n")
    lines.append(f"_Generated {date.today().isoformat()}._\n")

    n = results[0].n_examples if results else 0
    lines.append(f"**N**: {n} examples\n")

    # --- headline table ------------------------------------------------
    lines.append("## Headline")
    lines.append(
        "| baseline | acc | macro-F1 | ECE (10) | ECE (15) | Brier | NLL | stability (reorder, reword) | note |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        reorder = f"{r.stability_reorder:.3f}" if r.stability_reorder is not None else "n/a"
        lines.append(
            f"| {r.name} | {r.accuracy:.4f} | {r.macro_f1:.4f} | {r.ece_10:.4f} | "
            f"{r.ece_15:.4f} | {r.brier:.4f} | {r.nll:.4f} | "
            f"({reorder}, {r.stability_reword:.3f}) | {r.note} |"
        )
    lines.append("")

    # --- coverage sub-table ------------------------------------------
    lines.append("## Coverage")
    header_cols = " | ".join(f"@{int(c * 100)}%" for c in DEFAULT_COVERAGE_LEVELS)
    lines.append(f"| baseline | {header_cols} |")
    lines.append("|---|" + "---|" * len(DEFAULT_COVERAGE_LEVELS))
    for r in results:
        cells = " | ".join(f"{r.coverage[c]:.4f}" for c in DEFAULT_COVERAGE_LEVELS)
        lines.append(f"| {r.name} | {cells} |")
    lines.append("")

    # --- latency ------------------------------------------------------
    lines.append("## Per-call latency (one example at a time; not the serving-batched path)")
    lines.append("| baseline | p50 (ms) | p95 (ms) | mean (ms) |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.latency_ms['p50_ms']:.1f} | "
            f"{r.latency_ms['p95_ms']:.1f} | {r.latency_ms['mean_ms']:.1f} |"
        )
    lines.append("")

    # --- reproducibility footer --------------------------------------
    lines.append("## Reproducibility")
    for k, v in repro.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {"benchmark": benchmark_spec, "repro": repro, "results": [asdict(r) for r in results]},
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark", help='e.g. "academic/sst2" or "guardrails/wildguard_test"')
    ap.add_argument(
        "--baselines",
        default="bernoulli",
        help=f"comma list; valid names: {','.join(_KNOWN_BASELINES)}",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap eval examples")
    ap.add_argument(
        "--train-limit", type=int, default=None, help="cap train examples for trainable baselines"
    )
    ap.add_argument(
        "--bernoulli-url",
        default=os.environ.get("BERNOULLI_URL", "http://127.0.0.1:8000"),
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .md path; default = benchmarks/<spec>/results/<YYYY-MM-DD>.md",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover — invoked via CLI
    args = _parse_args(argv)

    module = _resolve_benchmark_module(args.benchmark)
    print(f"[run] loading {args.benchmark} ...", file=sys.stderr)
    examples = list(module.load(limit=args.limit))
    print(f"[run] {len(examples)} examples", file=sys.stderr)

    requested = [n.strip() for n in args.baselines.split(",") if n.strip()]
    for name in requested:
        if name not in _KNOWN_BASELINES:
            raise SystemExit(f"Unknown baseline {name!r}; must be one of {_KNOWN_BASELINES}")

    reword_stems: tuple[str, ...] = getattr(module, "REWORD_STEMS", ())
    train_examples: list[BenchmarkExample] | None = None
    needs_train = any(name == "bge-m3-lr" for name in requested)
    if needs_train:
        print("[run] loading training split for trainable baselines ...", file=sys.stderr)
        train_examples = list(module.load_train(limit=args.train_limit))
        print(f"[run] {len(train_examples)} training examples", file=sys.stderr)

    results: list[BaselineMetrics] = []
    for name in requested:
        print(f"[run] running baseline: {name}", file=sys.stderr)
        if name == "generative":
            baseline = _build_generative()
        else:
            baseline = _build_baseline(name, bernoulli_url=args.bernoulli_url, device=args.device)
        results.append(
            _run_one_baseline(
                name,
                baseline,
                examples,
                reword_stems=reword_stems,
                train_examples=train_examples,
                train_limit=args.train_limit,
            )
        )

    out = args.out
    if out is None:
        out = Path(module.__file__).parent / "results" / f"{date.today().isoformat()}.md"
    repro = _repro_footer(args.benchmark, module)
    write_report(out, args.benchmark, results, repro)
    print(f"[run] wrote {out}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
