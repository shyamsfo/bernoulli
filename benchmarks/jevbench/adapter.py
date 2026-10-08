"""BernoulliLocalAdapter — slot into the JevBench harness.

JevBench's upstream harness (`fstandhartinger/jevbench`, vendored at
`benchmarks/jevbench/upstream/`) ships adapter classes in
`jevbench.adapters.*` and registers them via a `kinds` dict in
`jevbench.cli.cmd_run`. Rather than fork upstream to add a new entry,
we provide our adapter here and register it from `benchmarks/jevbench/run.py`
before invoking the harness's `Runner` directly.

Why not use the shipped `openai_compat` adapter? It expects a
JSON-schema-constrained text completion and marks `probs_source =
"verbalized"`. Bernoulli produces NATIVE probabilities at a single
answer position — using the openai-compat path would misclassify
us as a text-and-parse system, which is specifically what Bernoulli
is NOT. We mark `probs_source = "native"`.

Mapping from JevBench task kinds → our `bernoulli.types.Question`:
- `noul` (binary) → `BinaryQuestion`. Our server returns
  `{"Yes": p, "No": 1-p}`; we lowercase to match JevBench's labels
  (`["no", "yes"]`).
- `choice` → `ChoiceQuestion`. Options are the canonical label order
  from `task.labels` (which are the keys of `task.question.criteria`).
- `score` → `RatingQuestion` with `scale = (0, len(criteria) - 1)`.
  Our server returns a distribution keyed by integer-strings matching
  `task.labels` exactly.

Cost: we run on our own hardware. Marginal cost per decision is zero
in the usual API sense; the amortised AWS g5.xlarge $1.01/hr spot is
reported in the submission description. The adapter emits `usage` with
latency so JevBench's Speed axis still has signal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The vendored upstream package isn't on sys.path by default — add it so
# `from jevbench.adapters.base import DecisionResult` works when this adapter
# is imported directly.
_UPSTREAM = Path(__file__).parent / "upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from jevbench.adapters.base import DecisionResult  # noqa: E402

from benchmarks.common.baselines import BernoulliHTTP  # noqa: E402
from benchmarks.common.dataset import BenchmarkExample  # noqa: E402
from bernoulli.types import BinaryQuestion, ChoiceQuestion, RatingQuestion  # noqa: E402


class BernoulliLocalAdapter:
    """JevBench adapter that calls our running `/v1/decide` server.

    Matches the informal adapter contract shared by every file in
    `jevbench/adapters/*.py`: a class with `name`, `cost_basis`, an
    `__init__` that accepts the keyword set the CLI passes in, and a
    `run(task) -> DecisionResult` method.
    """

    name = "bernoulli_local"
    cost_basis = "local_self_hosted_no_api_tariff"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        key_env: str = "",
        timeout_s: float = 120.0,
        price_input_per_m: float | None = None,
        price_output_per_m: float | None = None,
        revision: str | None = None,
        code_dir: str | None = None,
        **_: Any,  # swallow any extra kwargs the CLI may add
    ) -> None:
        self.endpoint = (endpoint or "http://127.0.0.1:8000").rstrip("/")
        self.model = model or "bernoulli-qwen-7b"
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.revision = revision
        self._client: BernoulliHTTP | None = None

    def _ensure_client(self) -> BernoulliHTTP:
        if self._client is None:
            self._client = BernoulliHTTP(base_url=self.endpoint, timeout=self.timeout_s)
        return self._client

    def reserve_estimate(self, task: Any) -> float:
        """No marginal API cost for a self-hosted server; the harness's budget
        ledger receives 0.0 per decision. Amortised hardware cost is reported
        separately in the submission description."""
        return 0.0

    def _build_example(self, task: Any) -> tuple[BenchmarkExample, str]:
        """Translate a JevBench task into a `BenchmarkExample` + a label-flavor tag.

        The flavor tag ("noul" | "choice" | "score") selects the output
        label-order normalisation in `_shape_probs`.
        """
        qtype = task.question["type"]
        criteria = task.question.get("criteria")
        instructions = task.question["instructions"]
        state_text = task.state if isinstance(task.state, str) else _json_dump(task.state)

        if qtype == "noul":
            # Binary. Our BinaryQuestion already uses Yes/No at the token level.
            q = BinaryQuestion(id="decision", prompt=instructions)
            return (
                BenchmarkExample(state_text=state_text, question=q, gold="", source="jevbench"),
                "noul",
            )

        if qtype == "choice":
            # task.labels is the canonical option-key order.
            options = [str(x) for x in task.labels]
            prompt = instructions
            if isinstance(criteria, dict):
                # Enrich the prompt with the per-option rubric so the model has
                # the full task description.
                rubric = "\n".join(f"- {k}: {criteria.get(k, k)}" for k in options)
                prompt = f"{instructions}\n\nOptions:\n{rubric}"
            q = ChoiceQuestion(id="decision", prompt=prompt, options=options)
            return (
                BenchmarkExample(state_text=state_text, question=q, gold="", source="jevbench"),
                "choice",
            )

        if qtype == "score":
            # Score levels are indexed [0, len(criteria)-1]. Map to a 1-based
            # RatingQuestion internally (bernoulli.types.RatingQuestion scale is
            # always positive integers), then shift the label keys back to 0-based
            # when returning probs.
            if not isinstance(criteria, list):
                raise TypeError(f"score task {task.id} has non-list criteria")
            n = len(criteria)
            # Pack the level descriptions into the prompt so the model sees them.
            legend = "\n".join(f"{i}: {d}" for i, d in enumerate(criteria))
            prompt = f"{instructions}\n\nLevels:\n{legend}"
            q = RatingQuestion(id="decision", prompt=prompt, scale=(1, n))
            return (
                BenchmarkExample(state_text=state_text, question=q, gold="", source="jevbench"),
                "score",
            )

        raise ValueError(f"unknown JevBench question type: {qtype!r}")

    def _shape_probs(self, probs: dict[str, float], flavor: str, task: Any) -> dict[str, float]:
        """Normalise our server's distribution keys to JevBench's canonical label order."""
        if flavor == "noul":
            # BernoulliHTTP returns {"Yes": p, "No": 1-p}. Lower-case + swap to label order.
            return {"no": float(probs.get("No", 0.0)), "yes": float(probs.get("Yes", 0.0))}
        if flavor == "choice":
            # Keys already match task.labels; preserve the canonical order.
            return {str(lab): float(probs.get(str(lab), 0.0)) for lab in task.labels}
        if flavor == "score":
            # Our RatingQuestion scale is 1-based ("1".."n"); JevBench uses "0".."n-1".
            return {str(i): float(probs.get(str(i + 1), 0.0)) for i in range(len(task.labels))}
        raise ValueError(flavor)

    def run(self, task: Any) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        try:
            example, flavor = self._build_example(task)
            client = self._ensure_client()
            import time

            t0 = time.perf_counter()
            raw_probs = client.predict(example)
            res.latency_s = time.perf_counter() - t0
            res.probs = self._shape_probs(raw_probs, flavor, task)
            res.ok = True
            res.usage = {"latency_s": res.latency_s}
        except Exception as e:
            res.error = f"{type(e).__name__}: {str(e)[:250]}"
        return res


def _json_dump(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
