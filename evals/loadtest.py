"""Load test the serving HTTP API.

Measures per-request latency over N requests at K questions per state. The
vision doc §7 Phase 3 pins the grid: K ∈ {1, 5, 20}. Default N = 50 per cell
— tight enough for a p50/p95 that stabilizes, cheap enough to run in a few
minutes against a single-GPU 7B.

Usage:
    python -m evals.loadtest --url http://127.0.0.1:8000 --out evals/reports/loadtest.md
    python -m evals.loadtest --url http://127.0.0.1:8000 --qks 1 5 20 --n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

_STATE_TEXT = (
    "Customer email: Where is my package? I ordered it two weeks ago and nothing has "
    "arrived yet. This is unacceptable service. I need an update right now or I'll "
    "file a chargeback. The tracking number was 1Z999AA10123456789 and the order "
    "number was 42-1138. Thanks."
)


def _questions(k: int) -> list[dict[str, object]]:
    """Fabricate k independent questions sharing the same state."""
    base: list[dict[str, object]] = [
        {
            "id": "intent",
            "type": "choice",
            "prompt": "What does the customer want?",
            "options": ["refund", "exchange", "tracking", "other"],
        },
        {"id": "urgent", "type": "binary", "prompt": "Is this urgent?"},
        {
            "id": "anger",
            "type": "rating",
            "prompt": "How angry is the customer?",
            "scale": [1, 5],
        },
        {
            "id": "sentiment",
            "type": "choice",
            "prompt": "Is the overall sentiment positive, neutral, or negative?",
            "options": ["positive", "neutral", "negative"],
        },
        {
            "id": "action_needed",
            "type": "choice",
            "prompt": "What action should support take?",
            "options": ["refund_now", "contact_customer", "escalate", "wait"],
        },
    ]
    # Repeat the base 5 enough times to cover k, giving each a unique id so pydantic
    # doesn't dedupe on validation elsewhere.
    out: list[dict[str, object]] = []
    i = 0
    while len(out) < k:
        q: dict[str, object] = {**base[i % len(base)]}
        q["id"] = f"{q['id']}_{i}"
        out.append(q)
        i += 1
    return out


@dataclass
class CellResult:
    questions_per_request: int
    n_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float


@dataclass
class LoadtestResult:
    url: str
    model: str
    revision: str | None
    cells: list[CellResult] = field(default_factory=list)
    warmup_requests: int = 0
    timestamp: str = ""


def _summarize(times_ms: list[float], k: int) -> CellResult:
    arr = sorted(times_ms)
    return CellResult(
        questions_per_request=k,
        n_requests=len(arr),
        p50_ms=statistics.median(arr),
        p95_ms=arr[int(len(arr) * 0.95)] if len(arr) > 1 else arr[0],
        p99_ms=arr[int(len(arr) * 0.99)] if len(arr) > 1 else arr[0],
        mean_ms=statistics.mean(arr),
        stdev_ms=statistics.pstdev(arr) if len(arr) > 1 else 0.0,
    )


def _run_cell(client: httpx.Client, url: str, k: int, n: int) -> list[float]:
    payload = {
        "state": {"text": _STATE_TEXT},
        "questions": _questions(k),
        "options": {"debias": "reverse", "calibrated": False},
    }
    times_ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post(url + "/v1/decide", json=payload)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        r.raise_for_status()
    return times_ms


def _fetch_model_info(client: httpx.Client, url: str) -> tuple[str, str | None]:
    r = client.get(url + "/v1/models")
    r.raise_for_status()
    m = r.json()["models"][0]
    return m["id"], m.get("revision")


def run(
    url: str,
    qks: list[int],
    n: int,
    warmup: int,
) -> LoadtestResult:
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        model, revision = _fetch_model_info(client, url)
        # Warm up — first calls include JIT compile for new shapes
        if warmup:
            _run_cell(client, url, max(qks), warmup)

        cells: list[CellResult] = []
        for k in qks:
            times_ms = _run_cell(client, url, k, n)
            cells.append(_summarize(times_ms, k))

    return LoadtestResult(
        url=url,
        model=model,
        revision=revision,
        cells=cells,
        warmup_requests=warmup,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _render(result: LoadtestResult) -> str:
    lines = [
        "# Load test report",
        "",
        f"- **Timestamp**: {result.timestamp}",
        f"- **URL**: `{result.url}`",
        f"- **Model**: `{result.model}` @ `{result.revision}`",
        f"- **Warmup requests**: {result.warmup_requests}",
        "",
        "## Per-request latency (ms)",
        "",
        "| questions/req | n | p50 | p95 | p99 | mean | stdev |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in result.cells:
        lines.append(
            f"| {c.questions_per_request} | {c.n_requests} | "
            f"{c.p50_ms:.1f} | {c.p95_ms:.1f} | {c.p99_ms:.1f} | "
            f"{c.mean_ms:.1f} | {c.stdev_ms:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loadtest")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--qks", type=int, nargs="+", default=[1, 5, 20], help="questions per request to sweep"
    )
    parser.add_argument("--n", type=int, default=50, help="requests per cell")
    parser.add_argument("--warmup", type=int, default=3, help="warmup requests before timing")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.url, args.qks, args.n, args.warmup)
    report = _render(result)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        args.out.with_suffix(".json").write_text(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
