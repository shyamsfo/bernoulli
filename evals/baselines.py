"""Generative baseline: same backbone, prompted for text, parsed.

The point of comparison. Bernoulli's logit path reads P(label token) at a
single answer position — one forward pass, calibrated-shape distribution.
A naive LLM user instead asks the model in natural language ("Which of
these options?") and parses whatever text comes back. This module is that
naive path, run with the same backbone so the only difference is the
reading mechanism.

Confidence modeling: the generative path has no native way to express
uncertainty — the model just produces a string. We assign probability 1.0
to the parsed answer and 0.0 to everything else (or a uniform distribution
if parsing fails). This exposes exactly how calibrated such a usage *isn't*;
the headline comparison for M3 is ECE / Brier / NLL between this and the
logit path.
"""

from __future__ import annotations

import time
from typing import Protocol, cast

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
)


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
