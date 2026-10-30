"""Run JevBench's harness against our local server with the Bernoulli adapter.

Usage (requires `just serve` already up at :8000):

    uv run python -m benchmarks.jevbench.run \
        --tasks benchmarks/jevbench/upstream/datasets/public/original.jsonl \
        --results benchmarks/jevbench/results/original-$(date +%F).jsonl \
        [--endpoint http://127.0.0.1:8000] \
        [--limit N]

This wraps `jevbench.runner.Runner` directly rather than patching the
upstream CLI's `kinds` dict. We import the vendored harness's primitives
(Runner + Ledger + task loader) and register `BernoulliLocalAdapter`
here. Keeps `benchmarks/jevbench/upstream/` read-only — no monkey-patch,
no fork.

After the run completes, pipe the results through the harness's
scoring + summarizer:

    uv run python -m jevbench.cli summarize \
        --tasks benchmarks/jevbench/upstream/datasets/public/original.jsonl \
        --results benchmarks/jevbench/results/original-<date>.jsonl \
        --ledger benchmarks/jevbench/results/original-<date>.ledger.json

That emits the four-axis Capability Score, which we then capture in
`benchmarks/jevbench/results/<date>.md` for a human-readable report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the vendored upstream package importable.
_UPSTREAM = Path(__file__).parent / "upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from jevbench.budget import Ledger  # noqa: E402
from jevbench.runner import DEFAULT_RESERVE_USD, Runner  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

from benchmarks.jevbench.adapter import BernoulliLocalAdapter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True, help="Path(s) to JSONL task file(s), comma-separated.")
    p.add_argument("--results", required=True, type=Path, help="Output JSONL path.")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000", help="Bernoulli server URL.")
    p.add_argument("--model", default="bernoulli-qwen-7b", help="Display name recorded in results.")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of tasks (smoke runs).")
    p.add_argument(
        "--cap-usd",
        type=float,
        default=1000.0,
        help="Budget ceiling for the Ledger. Self-hosted → set high; the ledger won't trip.",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Path to the budget ledger JSON. Defaults next to --results.",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory for raw per-task responses. Required by upstream Runner but "
            "defaults next to --results in `raw/<basename>/`. Should live outside "
            "the public repo for a real submission."
        ),
    )
    p.add_argument(
        "--delay-s",
        type=float,
        default=0.0,
        help="Delay between tasks. 0 is fine for local; raise for API rate-limited systems.",
    )
    return p.parse_args()


def _load_all(spec: str) -> list:
    tasks = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            tasks.extend(load_jsonl(part))
    return tasks


def main() -> int:
    args = _parse_args()

    tasks = _load_all(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    adapter = BernoulliLocalAdapter(endpoint=args.endpoint, model=args.model)
    ledger_path = args.ledger or args.results.with_suffix(".ledger.json")
    ledger = Ledger(str(ledger_path), cap_usd=args.cap_usd)
    # Upstream Runner demands a non-None raw_dir AND refuses to write inside
    # the vendored upstream/ tree. Default to `raw/<results-basename>/` next
    # to the results file; this is both outside upstream/ and inside our repo
    # (ignored via benchmarks/jevbench/.gitignore so raw blobs don't get
    # committed by accident).
    raw_dir = args.raw_dir or (args.results.parent / "raw" / args.results.stem)
    raw_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(
        adapter,
        ledger,
        raw_dir=str(raw_dir),
        default_reserve_usd=DEFAULT_RESERVE_USD,
    )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[jevbench] adapter={adapter.name} endpoint={adapter.endpoint} "
        f"tasks={len(tasks)} cap=${args.cap_usd:.0f} ledger={ledger_path}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    records = runner.run_all(tasks, results_path=str(args.results), delay_s=args.delay_s)
    elapsed = time.perf_counter() - t0
    failed = sum(1 for r in records if r["status"] == "failed")
    print(
        f"[jevbench] done: {len(records)}/{len(tasks)} attempted, {failed} failed; "
        f"wall={elapsed:.1f}s; charged=${ledger.charged:.4f}",
        file=sys.stderr,
    )
    return 0 if len(records) == len(tasks) else 3


if __name__ == "__main__":  # pragma: no cover — CLI
    sys.exit(main())
