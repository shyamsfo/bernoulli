"""The common `BenchmarkExample` format every benchmark loader produces.

Each loader in `benchmarks.<category>.<name>` exposes:

    def load(limit: int | None = None) -> Iterator[BenchmarkExample]: ...

The runner iterates these, calls `/v1/decide` (or the configured baseline)
per example, and feeds the predicted distribution + gold into
`benchmarks.common.metrics`.

Relation to `evals.example.EvalExample`: BenchmarkExample is a drop-in
superset — same four core fields with the same names + a `meta` dict for
benchmark-specific extras (example_id for seed-pinning, is_oos for
CLINC150, publication_date for arXiv post-cutoff, etc.). Migrations
from `evals/datasets/*.py` into `benchmarks/<category>/<name>/` are
mechanical: rename the import, optionally populate `meta`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bernoulli.types import Question


@dataclass(frozen=True)
class BenchmarkExample:
    """One benchmark example in the common format.

    - `gold` is the option string the model should pick, matched to the
      response distribution keys:
        * choice: one of `question.options`
        * binary: 'Yes' or 'No'
        * rating: integer label as string, e.g. '3' for scale (1, 5)
    - `source` is the benchmark tag (e.g. 'sst2', 'wildguard_test') used
      in reports and the leaderboard table.
    - `meta` carries benchmark-specific extras. Conventions by benchmark:
        * `example_id: str` — stable id for seed-pinning (any benchmark)
        * `is_oos: bool`    — out-of-scope flag (clinc150_oos)
        * `publication_date: str` — ISO date (arxiv_post_cutoff)
        * `category: str`   — fine-grained class for stratified analysis
      Keep it flat and JSON-serialisable so results sidecars round-trip.

    `meta` is a dict for convenience; treat it as immutable after
    construction (the outer dataclass is frozen but the dict is not —
    don't mutate it in consumers).
    """

    state_text: str
    question: Question
    gold: str
    source: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)
