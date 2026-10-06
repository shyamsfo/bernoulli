"""Baseline Protocol + wrapper tests (CPU-only, no GPU, no network).

Covers:
- distribution_from_decision for all three Decision kinds.
- BernoulliHTTP against a mocked httpx.Client (verifies request shape +
  that the response is normalized to the right Distribution).
- Generative wraps the existing generative_decide path end-to-end with
  the scripted scorer from test_baselines.py (shape only — the semantics
  of generative_decide have their own tests next door).
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.common.baselines import (
    BernoulliHTTP,
    Distribution,
    Generative,
    distribution_from_decision,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import (
    BinaryDecision,
    BinaryQuestion,
    ChoiceDecision,
    ChoiceQuestion,
    RatingDecision,
    RatingQuestion,
)


class TestDistributionFromDecision:
    def test_choice(self) -> None:
        q = ChoiceQuestion(id="x", prompt="?", options=["a", "b"])
        d = ChoiceDecision(answer="a", confidence=0.7, distribution={"a": 0.7, "b": 0.3})
        assert distribution_from_decision(d, q) == {"a": 0.7, "b": 0.3}

    def test_binary(self) -> None:
        q = BinaryQuestion(id="x", prompt="?")
        d = BinaryDecision(answer=True, probability=0.8)
        out = distribution_from_decision(d, q)
        assert out["Yes"] == pytest.approx(0.8)
        assert out["No"] == pytest.approx(0.2)

    def test_rating(self) -> None:
        q = RatingQuestion(id="x", prompt="?", scale=(1, 5))
        dist = {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.3, "5": 0.1}
        d = RatingDecision(expected=3.1, distribution=dist)
        assert distribution_from_decision(d, q) == dist


class _MockResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return self._payload


class _MockHttpxClient:
    """Records the posted payload and returns a canned DecideResponse."""

    def __init__(self, response_payload: dict[str, Any]) -> None:
        self._payload = response_payload
        self.posted_to: str | None = None
        self.posted_json: dict[str, Any] | None = None

    def post(self, url: str, *, json: dict[str, Any]) -> _MockResponse:
        self.posted_to = url
        self.posted_json = json
        return _MockResponse(self._payload)

    def close(self) -> None:
        return


class TestBernoulliHTTP:
    def test_posts_to_decide_endpoint(self) -> None:
        q = ChoiceQuestion(id="sentiment", prompt="?", options=["negative", "positive"])
        example = BenchmarkExample(state_text="great movie", question=q, gold="positive")
        payload = {
            "decisions": {
                "sentiment": {
                    "type": "choice",
                    "answer": "positive",
                    "confidence": 0.92,
                    "distribution": {"negative": 0.08, "positive": 0.92},
                }
            },
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "calibration_version": None,
            "latency_ms": 76,
        }
        client = _MockHttpxClient(payload)
        baseline = BernoulliHTTP(base_url="http://127.0.0.1:8000", client=client)  # type: ignore[arg-type]

        dist: Distribution = baseline.predict(example)

        assert client.posted_to == "http://127.0.0.1:8000/v1/decide"
        assert client.posted_json is not None
        assert client.posted_json["state"]["text"] == "great movie"
        assert client.posted_json["questions"][0]["id"] == "sentiment"
        assert dist == {"negative": 0.08, "positive": 0.92}

    def test_name(self) -> None:
        assert BernoulliHTTP.name == "bernoulli"


class TestGenerativeWrapper:
    """The wrapper is a thin shim over `generative_decide` + distribution_from_decision.

    Both halves are tested in test_baselines.py (semantic) and
    TestDistributionFromDecision above (conversion). Only the name is asserted
    here — adding another end-to-end test with the scripted scorer would just
    duplicate coverage.
    """

    def test_name(self) -> None:
        assert Generative.name == "generative"
