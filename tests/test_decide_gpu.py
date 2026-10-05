"""End-to-end decide() tests against the real backbone. GPU-marked.

These are the real correctness tests. Everything else is types + shape.
"""

from __future__ import annotations

import math

import pytest

from bernoulli.decide import decide
from bernoulli.types import (
    BinaryQuestion,
    ChoiceQuestion,
    DecideRequest,
    RatingQuestion,
    State,
)

pytestmark = pytest.mark.gpu


def _sums_to_one(dist: dict[str, float]) -> bool:
    return math.isclose(sum(dist.values()), 1.0, abs_tol=1e-4)


def test_choice_identifies_refund(hf_scorer: object) -> None:
    req = DecideRequest(
        state=State(text="Customer email: I want my money back now!"),
        questions=[
            ChoiceQuestion(
                id="intent",
                prompt="What does the customer want?",
                options=["refund", "exchange", "tracking", "other"],
            ),
        ],
    )
    resp = decide(req, hf_scorer)  # type: ignore[arg-type]
    d = resp.decisions["intent"]
    assert d.type == "choice"
    assert d.answer == "refund"
    assert d.confidence > 0.5
    assert _sums_to_one(d.distribution)
    assert set(d.distribution) == {"refund", "exchange", "tracking", "other"}


def test_binary_urgent(hf_scorer: object) -> None:
    req = DecideRequest(
        state=State(text="PATIENT COLLAPSING. SEND HELP NOW."),
        questions=[BinaryQuestion(id="urgent", prompt="Is this urgent?")],
    )
    resp = decide(req, hf_scorer)  # type: ignore[arg-type]
    d = resp.decisions["urgent"]
    assert d.type == "binary"
    assert d.answer is True
    assert d.probability > 0.5


def test_rating_angry(hf_scorer: object) -> None:
    req = DecideRequest(
        state=State(text="I AM FURIOUS. THIS IS THE THIRD TIME I HAVE EMAILED YOU. UNACCEPTABLE."),
        questions=[RatingQuestion(id="anger", prompt="How angry is the customer?", scale=(1, 5))],
    )
    resp = decide(req, hf_scorer)  # type: ignore[arg-type]
    d = resp.decisions["anger"]
    assert d.type == "rating"
    assert _sums_to_one(d.distribution)
    assert set(d.distribution) == {"1", "2", "3", "4", "5"}
    assert 3.0 < d.expected <= 5.0  # a furious customer should land in the upper half


def test_multi_question_shares_state(hf_scorer: object) -> None:
    req = DecideRequest(
        state=State(text="I want my money back. This is unacceptable."),
        questions=[
            ChoiceQuestion(
                id="intent", prompt="What do they want?", options=["refund", "info", "other"]
            ),
            BinaryQuestion(id="urgent", prompt="Is this urgent?"),
            RatingQuestion(id="anger", prompt="Anger level?", scale=(1, 5)),
        ],
    )
    resp = decide(req, hf_scorer)  # type: ignore[arg-type]
    assert set(resp.decisions) == {"intent", "urgent", "anger"}
    assert resp.decisions["intent"].type == "choice"
    assert resp.decisions["urgent"].type == "binary"
    assert resp.decisions["anger"].type == "rating"
    assert resp.model.startswith("Qwen/")
    assert resp.latency_ms >= 0


def test_deterministic(hf_scorer: object) -> None:
    """Two runs of the same request should give identical distributions.

    This is the test the vision doc §10 asked for ('every module gets tests;
    eval numbers come from scripts, never from hand-typed results')."""
    req = DecideRequest(
        state=State(text="Where is my package?"),
        questions=[
            ChoiceQuestion(
                id="intent", prompt="?", options=["refund", "exchange", "tracking", "other"]
            ),
        ],
    )
    a = decide(req, hf_scorer).decisions["intent"]  # type: ignore[arg-type]
    b = decide(req, hf_scorer).decisions["intent"]  # type: ignore[arg-type]
    for key in a.distribution:
        assert math.isclose(a.distribution[key], b.distribution[key], abs_tol=1e-5)
