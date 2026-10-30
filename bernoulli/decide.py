"""Dispatcher: Request + Scorer -> Response.

Threads the scorer, debiaser, and type-specific decision assembly in one
place. When a `Calibration` is supplied and `request.options.calibrated`
is True, the averaged per-option probabilities are T-scaled before the
Decision is assembled.
"""

from __future__ import annotations

import time

import numpy as np

from bernoulli.calibrate import Calibration, apply_temperature
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


def _question_type(question: Question) -> str:
    if isinstance(question, ChoiceQuestion):
        return "choice"
    if isinstance(question, BinaryQuestion):
        return "binary"
    return "rating"


def _decide_one(
    scorer: Scorer,
    state_text: str,
    question: Question,
    debias_mode: str,
    calibration: Calibration | None,
) -> Decision:
    if isinstance(question, ChoiceQuestion) and len(question.options) > CHUNK_SIZE:
        # Many-option choice: debias is deferred (see chunked.py). Mode is ignored.
        probs = chunked_choice(scorer, state_text=state_text, question=question)
    else:
        probs = debias(scorer, state_text=state_text, question=question, mode=debias_mode)  # type: ignore[arg-type]

    options = option_strings_for(question)
    probs_dict: dict[str, float] = {opt: float(p) for opt, p in zip(options, probs, strict=True)}

    if calibration is not None:
        t = calibration.t_for(_question_type(question))  # type: ignore[arg-type]
        if t != 1.0:
            probs_dict = apply_temperature(probs_dict, t)

    if isinstance(question, ChoiceQuestion):
        best_opt = max(probs_dict, key=lambda k: probs_dict[k])
        return ChoiceDecision(
            answer=best_opt,
            confidence=float(probs_dict[best_opt]),
            distribution=probs_dict,
        )

    if isinstance(question, BinaryQuestion):
        p_yes = float(probs_dict["Yes"])
        return BinaryDecision(answer=p_yes >= 0.5, probability=p_yes)

    if isinstance(question, RatingQuestion):
        # distribution keys are integer-string labels; expected = Σ p_i · value_i
        values = np.array([int(k) for k in probs_dict], dtype=np.float64)
        p_arr = np.array(list(probs_dict.values()), dtype=np.float64)
        return RatingDecision(
            expected=float((p_arr * values).sum()),
            distribution=probs_dict,
        )

    raise TypeError(f"unknown question kind: {type(question).__name__}")


def decide(
    request: DecideRequest,
    scorer: Scorer,
    *,
    calibration: Calibration | None = None,
) -> DecideResponse:
    """Run every question in the request against the scorer.

    If `calibration` is supplied and `request.options.calibrated` is True,
    T-scale per question type before assembling the response. Otherwise the
    calibration argument is ignored.
    """
    if request.state.images:
        raise NotImplementedError("image states land in M4")

    use_cal = calibration if request.options.calibrated else None

    t0 = time.perf_counter()
    decisions: dict[str, Decision] = {}
    for q in request.questions:
        decisions[q.id] = _decide_one(
            scorer, request.state.text, q, request.options.debias, use_cal
        )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return DecideResponse(
        decisions=decisions,
        model=scorer.model_id,
        calibration_version=use_cal.version if use_cal is not None else None,
        latency_ms=elapsed_ms,
    )
