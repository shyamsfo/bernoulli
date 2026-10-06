"""Shared baselines for every benchmark.

Two layers:

- **High-level `Baseline` protocol** — the shape the runner consumes.
  Every baseline wraps the per-example call in `predict(example) ->
  Distribution`. The runner iterates benchmark examples and calls each
  baseline once per example, feeding the distributions into
  `benchmarks.common.metrics`.

- **Low-level primitives** — `generative_decide()` and
  `parse_response()` are the same-backbone text-and-parse implementation
  originally built in M3. Kept here because `Generative` wraps them;
  also usable directly for M3 compatibility.

Baselines in this module:

- `BernoulliHTTP` — POST to a running `/v1/decide` server. The honest
  shipping path.
- `Generative` — same backbone, prompted for text, parsed. The ECE/NLL
  comparison that isolates the logit vs. generated-token gap. Direct
  (in-process via HFScorer) because the server does not expose a
  generative endpoint today; moving this to HTTP is a server change,
  not a baseline change.

DeBERTa-v3-zeroshot and BGE-m3 + logistic-regression baselines are
their own sub-tasks (M8 task 4b, 4c) — each pulls in a new HF model
and makes most sense in its own commit alongside its dependency bump.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from bernoulli.prompt import option_strings_for
from bernoulli.types import (
    BinaryDecision,
    BinaryQuestion,
    ChoiceDecision,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    Decision,
    Question,
    RatingDecision,
    RatingQuestion,
    State,
)

if TYPE_CHECKING:
    import httpx

    from benchmarks.common.dataset import BenchmarkExample


class _HasGenerate(Protocol):
    """Structural type: anything with .generate(prompt, max_new_tokens) -> str + .tokenizer + .model_id."""

    tokenizer: object
    model_id: str

    def generate(self, prompt: str, *, max_new_tokens: int = ...) -> str: ...


def _build_generative_prompt(tok: object, state_text: str, question: Question) -> str:
    """Prompt a classifier-shaped completion: options listed in system, state + question in user."""
    options = option_strings_for(question)
    options_fmt = ", ".join(options)
    system = (
        "You are a classifier. Answer with exactly one of these options and nothing else: "
        f"{options_fmt}."
    )
    user = f"{state_text}\n\n{question.prompt}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    rendered = tok.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    return cast(str, rendered)


def parse_response(text: str, options: list[str]) -> int | None:
    """Return the index of the option found earliest in the response, or None.

    Case-insensitive. First match by position wins (so "refund" wins over
    "other" if both appear and refund appears first). None signals a parse
    failure, which the caller handles by falling back to a uniform distribution.
    """
    normalized = text.lower()
    hits = []
    for i, opt in enumerate(options):
        pos = normalized.find(opt.lower())
        if pos >= 0:
            hits.append((pos, i))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def _probs_from_choice(chosen: int | None, n: int) -> NDArray[np.float32]:
    probs = np.zeros(n, dtype=np.float32)
    if chosen is None:
        probs[:] = 1.0 / n  # uniform fallback on parse failure
    else:
        probs[chosen] = 1.0
    return probs


def _decide_one_generative(
    scorer: _HasGenerate,
    state_text: str,
    question: Question,
    *,
    max_new_tokens: int = 10,
) -> Decision:
    options = option_strings_for(question)
    prompt = _build_generative_prompt(scorer.tokenizer, state_text, question)
    text = scorer.generate(prompt, max_new_tokens=max_new_tokens)
    chosen = parse_response(text, options)
    probs = _probs_from_choice(chosen, len(options))

    if isinstance(question, ChoiceQuestion):
        best_idx = int(np.argmax(probs))
        return ChoiceDecision(
            answer=options[best_idx],
            confidence=float(probs[best_idx]),
            distribution={opt: float(p) for opt, p in zip(options, probs, strict=True)},
        )

    if isinstance(question, BinaryQuestion):
        p_yes = float(probs[0])  # labels_for returns ['Yes', 'No']
        return BinaryDecision(answer=p_yes >= 0.5, probability=p_yes)

    if isinstance(question, RatingQuestion):
        low, _ = question.scale
        values = np.arange(low, low + len(probs), dtype=np.float64)
        return RatingDecision(
            expected=float((probs * values).sum()),
            distribution={str(v): float(p) for v, p in zip(values.astype(int), probs, strict=True)},
        )

    raise TypeError(f"unknown question kind: {type(question).__name__}")


def generative_decide(request: DecideRequest, scorer: _HasGenerate) -> DecideResponse:
    """Baseline dispatcher: for each question, generate a short answer and parse it."""
    if request.state.images:
        raise NotImplementedError("image states land in M4")

    t0 = time.perf_counter()
    decisions: dict[str, Decision] = {}
    for q in request.questions:
        decisions[q.id] = _decide_one_generative(scorer, request.state.text, q)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return DecideResponse(
        decisions=decisions,
        model=scorer.model_id,
        calibration_version=None,
        latency_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# High-level Baseline protocol + wrappers consumed by the benchmark runner.
# ---------------------------------------------------------------------------

Distribution = dict[str, float]


class Baseline(Protocol):
    """Common shape every baseline implements.

    `name` is the short, stable column id used in the leaderboard table
    (`bernoulli`, `generative`, `deberta`, `bge-m3`). `predict` is called
    once per example; baselines own their own inference state (loaded
    model, HTTP client) and are constructed once per benchmark run.
    """

    name: str

    def predict(self, example: BenchmarkExample) -> Distribution: ...


def distribution_from_decision(decision: Decision, question: Question) -> Distribution:
    """Flatten a Decision into the {label_str: prob} shape the metrics module expects.

    - ChoiceDecision: already a distribution over option strings.
    - BinaryDecision: {'Yes': p, 'No': 1 - p}.
    - RatingDecision: already a distribution over integer-string keys.
    """
    if isinstance(decision, ChoiceDecision):
        return {k: float(v) for k, v in decision.distribution.items()}
    if isinstance(decision, BinaryDecision):
        return {"Yes": float(decision.probability), "No": float(1.0 - decision.probability)}
    if isinstance(decision, RatingDecision):
        return {k: float(v) for k, v in decision.distribution.items()}
    raise TypeError(f"unknown decision kind: {type(decision).__name__}")


class BernoulliHTTP:
    """Bernoulli over HTTP against a running `/v1/decide` server.

    One request per example — the server batches `questions x permutations`
    internally. Baselines are expected to be slower than production serving
    (per-example calls, no multi-example batching), which is honest for
    like-for-like comparison with the other baselines in this module.
    """

    name = "bernoulli"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        import httpx as _httpx

        self._client = client if client is not None else _httpx.Client(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    def predict(self, example: BenchmarkExample) -> Distribution:
        payload = {
            "state": {"text": example.state_text},
            "questions": [example.question.model_dump()],
        }
        r = self._client.post(f"{self._base_url}/v1/decide", json=payload)
        r.raise_for_status()
        resp = DecideResponse.model_validate(r.json())
        return distribution_from_decision(resp.decisions[example.question.id], example.question)

    def close(self) -> None:
        self._client.close()


class Generative:
    """Same-backbone generative baseline — direct, in-process via `HFScorer`.

    Wraps `generative_decide` above. Direct because the server does not
    expose a generative endpoint today; the comparison a benchmark cares
    about is backbone + reading-mechanism, not transport.
    """

    name = "generative"

    def __init__(self, scorer: _HasGenerate) -> None:
        self._scorer = scorer

    def predict(self, example: BenchmarkExample) -> Distribution:
        req = DecideRequest(
            state=State(text=example.state_text),
            questions=[example.question],
        )
        resp = generative_decide(req, self._scorer)
        return distribution_from_decision(resp.decisions[example.question.id], example.question)
