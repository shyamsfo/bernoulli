"""Type-level smoke tests for the API spec.

Full semantic tests (option-string round-trip, 422 on invalid, deterministic
outputs, distribution sums to 1) land in M2 when the scorer exists.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bernoulli.types import (
    BinaryQuestion,
    ChoiceQuestion,
    DecideOptions,
    DecideRequest,
    DecideResponse,
    RatingQuestion,
    State,
)


class TestState:
    def test_empty_is_valid(self) -> None:
        State()

    def test_text_only(self) -> None:
        s = State(text="hello")
        assert s.text == "hello"
        assert s.images == []

    def test_max_eight_images(self) -> None:
        State(images=["a"] * 8)
        with pytest.raises(ValidationError):
            State(images=["a"] * 9)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            State(text="x", unknown_field="bad")  # type: ignore[call-arg]


class TestChoiceQuestion:
    def test_valid(self) -> None:
        q = ChoiceQuestion(id="intent", prompt="what?", options=["a", "b", "c"])
        assert q.type == "choice"

    def test_rejects_one_option(self) -> None:
        with pytest.raises(ValidationError):
            ChoiceQuestion(id="x", prompt="?", options=["only"])

    def test_rejects_more_than_128_options(self) -> None:
        with pytest.raises(ValidationError):
            ChoiceQuestion(id="x", prompt="?", options=[f"o{i}" for i in range(129)])

    def test_allows_up_to_128_options(self) -> None:
        """Chunked scoring handles the 26-letter alphabet overflow; 128 is the current cap."""
        ChoiceQuestion(id="x", prompt="?", options=[f"o{i}" for i in range(128)])


class TestBinaryQuestion:
    def test_valid(self) -> None:
        q = BinaryQuestion(id="urgent", prompt="is this urgent?")
        assert q.type == "binary"


class TestRatingQuestion:
    def test_valid(self) -> None:
        q = RatingQuestion(id="anger", prompt="how angry?", scale=(1, 5))
        assert q.type == "rating"
        assert q.scale == (1, 5)

    def test_rejects_inverted_scale(self) -> None:
        with pytest.raises(ValidationError):
            RatingQuestion(id="x", prompt="?", scale=(5, 1))

    def test_rejects_too_wide_scale(self) -> None:
        with pytest.raises(ValidationError):
            RatingQuestion(id="x", prompt="?", scale=(1, 10))  # width 10 > 9


class TestDecideRequest:
    def test_full_shape(self) -> None:
        req = DecideRequest(
            state=State(text="Customer email body..."),
            questions=[
                ChoiceQuestion(
                    id="intent",
                    prompt="What does the customer want?",
                    options=["refund", "exchange", "tracking", "other"],
                ),
                BinaryQuestion(id="urgent", prompt="Is this message urgent?"),
                RatingQuestion(id="anger", prompt="How angry?", scale=(1, 5)),
            ],
            options=DecideOptions(debias="reverse", calibrated=True),
        )
        assert len(req.questions) == 3
        assert req.options.debias == "reverse"

    def test_default_options(self) -> None:
        req = DecideRequest(
            state=State(text="x"),
            questions=[BinaryQuestion(id="q", prompt="?")],
        )
        assert req.options.debias == "reverse"
        assert req.options.calibrated is True

    def test_requires_at_least_one_question(self) -> None:
        with pytest.raises(ValidationError):
            DecideRequest(state=State(text="x"), questions=[])

    def test_discriminated_union_from_dict(self) -> None:
        """Confirm pydantic routes dict payloads by `type` to the right class."""
        req = DecideRequest.model_validate(
            {
                "state": {"text": "x"},
                "questions": [
                    {"id": "a", "type": "choice", "prompt": "?", "options": ["x", "y"]},
                    {"id": "b", "type": "binary", "prompt": "?"},
                    {"id": "c", "type": "rating", "prompt": "?", "scale": [1, 3]},
                ],
            }
        )
        assert [q.type for q in req.questions] == ["choice", "binary", "rating"]


class TestDecideResponse:
    def test_shape(self) -> None:
        resp = DecideResponse.model_validate(
            {
                "decisions": {
                    "intent": {
                        "type": "choice",
                        "answer": "refund",
                        "confidence": 0.91,
                        "distribution": {"refund": 0.91, "other": 0.09},
                    },
                    "urgent": {
                        "type": "binary",
                        "answer": True,
                        "probability": 0.78,
                    },
                    "anger": {
                        "type": "rating",
                        "expected": 3.6,
                        "distribution": {"1": 0.02, "2": 0.08, "3": 0.25, "4": 0.48, "5": 0.17},
                    },
                },
                "model": "qwen2.5-vl-7b-instruct",
                "calibration_version": "2026-10-05",
                "latency_ms": 142,
            }
        )
        assert resp.decisions["intent"].type == "choice"
        assert resp.decisions["urgent"].type == "binary"
        assert resp.decisions["anger"].type == "rating"
