"""Dispatcher: Request + Scorer -> Response.

Threads the scorer, debiaser, and type-specific decision assembly in one
place. Calibration (M3+) slots in by scaling the averaged probabilities with
a per-question-type temperature before the Decision is assembled; that path
is wired here when calibrate.py lands.
"""

from __future__ import annotations

import time

import numpy as np

from bernoulli.chunked import CHUNK_SIZE, chunked_choice
from bernoulli.debias import debias
from bernoulli.prompt import option_strings_for
from bernoulli.scorer import Scorer
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
)


def _decide_one(scorer: Scorer, state_text: str, question: Question, debias_mode: str) -> Decision:
    if isinstance(question, ChoiceQuestion) and len(question.options) > CHUNK_SIZE:
        # Many-option choice: debias is deferred (see chunked.py). Mode is ignored.
        probs = chunked_choice(scorer, state_text=state_text, question=question)
    else:
        probs = debias(scorer, state_text=state_text, question=question, mode=debias_mode)  # type: ignore[arg-type]

    options = option_strings_for(question)

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


def decide(request: DecideRequest, scorer: Scorer) -> DecideResponse:
    """Run every question in the request against the scorer. Returns typed decisions."""
    if request.state.images:
        raise NotImplementedError("image states land in M4")

    t0 = time.perf_counter()
    decisions: dict[str, Decision] = {}
    for q in request.questions:
        decisions[q.id] = _decide_one(scorer, request.state.text, q, request.options.debias)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return DecideResponse(
        decisions=decisions,
        model=scorer.model_id,
        calibration_version=None,
        latency_ms=elapsed_ms,
    )
