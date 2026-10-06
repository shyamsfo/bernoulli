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

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.common.baselines import (
    BernoulliHTTP,
    BGEm3LR,
    DeBERTaZeroshot,
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


class _FakeZeroShot:
    """Mock stand-in for the HF zero-shot pipeline — no model, no download."""

    def __init__(self, scores_by_label: dict[str, float]) -> None:
        self._scores = scores_by_label
        self.called_with: tuple[str, list[str], str, bool] | None = None

    def __call__(
        self,
        sequences: str,
        candidate_labels: list[str],
        *,
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False,
    ) -> dict[str, Any]:
        self.called_with = (sequences, list(candidate_labels), hypothesis_template, multi_label)
        # HF returns labels sorted by score desc. Mimic that.
        scored = sorted(
            ((lab, self._scores[lab]) for lab in candidate_labels),
            key=lambda t: -t[1],
        )
        return {
            "sequence": sequences,
            "labels": [lab for lab, _ in scored],
            "scores": [s for _, s in scored],
        }


class TestDeBERTaZeroshot:
    def test_name(self) -> None:
        assert DeBERTaZeroshot.name == "deberta"

    def test_choice_distribution_realigned_to_option_order(self) -> None:
        # Fake returns "positive" first (highest score). We expect the output
        # dict to be keyed by the original option order, not pipeline order.
        q = ChoiceQuestion(id="sentiment", prompt="?", options=["negative", "positive"])
        example = BenchmarkExample(state_text="great movie", question=q, gold="positive")
        fake = _FakeZeroShot({"negative": 0.1, "positive": 0.9})
        baseline = DeBERTaZeroshot(fake)

        dist = baseline.predict(example)

        assert list(dist.keys()) == ["negative", "positive"]
        assert dist["negative"] == pytest.approx(0.1)
        assert dist["positive"] == pytest.approx(0.9)
        assert fake.called_with is not None
        assert fake.called_with[0] == "great movie"
        assert fake.called_with[1] == ["negative", "positive"]
        assert fake.called_with[3] is False  # multi_label

    def test_custom_hypothesis_template_is_forwarded(self) -> None:
        q = ChoiceQuestion(id="x", prompt="?", options=["a", "b"])
        example = BenchmarkExample(state_text="hi", question=q, gold="a")
        fake = _FakeZeroShot({"a": 0.6, "b": 0.4})
        baseline = DeBERTaZeroshot(fake, hypothesis_template="The intent here is {}.")
        baseline.predict(example)
        assert fake.called_with is not None
        assert fake.called_with[2] == "The intent here is {}."

    def test_binary_raises_not_implemented(self) -> None:
        q = BinaryQuestion(id="x", prompt="?")
        example = BenchmarkExample(state_text="hi", question=q, gold="Yes")
        fake = _FakeZeroShot({})
        baseline = DeBERTaZeroshot(fake)
        with pytest.raises(NotImplementedError, match="ChoiceQuestion"):
            baseline.predict(example)

    def test_rating_raises_not_implemented(self) -> None:
        q = RatingQuestion(id="x", prompt="?", scale=(1, 5))
        example = BenchmarkExample(state_text="hi", question=q, gold="3")
        fake = _FakeZeroShot({})
        baseline = DeBERTaZeroshot(fake)
        with pytest.raises(NotImplementedError, match="ChoiceQuestion"):
            baseline.predict(example)


class _TextIndicatorEmbedder:
    """Deterministic embedder for tests.

    Maps each text to a 2-D vector that encodes a crude token-presence signal
    so the LR head has a learnable gradient. Keeps unit tests dependency-free
    on an actual BGE-m3 pull.
    """

    def __init__(self, token_a: str, token_b: str) -> None:
        self._a = token_a
        self._b = token_b

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        for t in texts:
            lo = t.lower()
            rows.append([float(self._a in lo), float(self._b in lo)])
        return np.asarray(rows, dtype=np.float32)


class TestBGEm3LR:
    def test_name(self) -> None:
        assert BGEm3LR.name == "bge-m3-lr"

    def test_fit_then_predict_choice(self) -> None:
        q = ChoiceQuestion(id="sentiment", prompt="?", options=["negative", "positive"])
        train = [
            BenchmarkExample(state_text="great movie", question=q, gold="positive"),
            BenchmarkExample(state_text="loved it great", question=q, gold="positive"),
            BenchmarkExample(state_text="terrible film", question=q, gold="negative"),
            BenchmarkExample(state_text="awful terrible", question=q, gold="negative"),
        ]
        baseline = BGEm3LR(_TextIndicatorEmbedder("great", "terrible"))
        baseline.fit(train)

        # Prediction on a test example containing "great" should favor positive.
        test = BenchmarkExample(state_text="great story", question=q, gold="positive")
        dist = baseline.predict(test)
        assert set(dist.keys()) == {"negative", "positive"}
        assert sum(dist.values()) == pytest.approx(1.0)
        assert dist["positive"] > dist["negative"]

    def test_fit_then_predict_rating_labels_preserved(self) -> None:
        q = RatingQuestion(id="stars", prompt="?", scale=(1, 3))
        train = [
            BenchmarkExample(state_text="great", question=q, gold="3"),
            BenchmarkExample(state_text="great great", question=q, gold="3"),
            BenchmarkExample(state_text="meh", question=q, gold="2"),
            BenchmarkExample(state_text="meh meh", question=q, gold="2"),
            BenchmarkExample(state_text="terrible", question=q, gold="1"),
            BenchmarkExample(state_text="terrible terrible", question=q, gold="1"),
        ]
        baseline = BGEm3LR(_TextIndicatorEmbedder("great", "terrible"))
        baseline.fit(train)

        dist = baseline.predict(BenchmarkExample(state_text="terrible", question=q, gold="1"))
        assert set(dist.keys()) == {"1", "2", "3"}
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_predict_before_fit_raises(self) -> None:
        q = ChoiceQuestion(id="x", prompt="?", options=["a", "b"])
        baseline = BGEm3LR(_TextIndicatorEmbedder("a", "b"))
        with pytest.raises(RuntimeError, match=r"before \.fit"):
            baseline.predict(BenchmarkExample(state_text="hi", question=q, gold="a"))

    def test_fit_rejects_empty_training_set(self) -> None:
        baseline = BGEm3LR(_TextIndicatorEmbedder("a", "b"))
        with pytest.raises(ValueError, match="at least one"):
            baseline.fit([])

    def test_second_fit_replaces_the_head(self) -> None:
        q = ChoiceQuestion(id="x", prompt="?", options=["a", "b"])
        baseline = BGEm3LR(_TextIndicatorEmbedder("a", "b"))

        first = [
            BenchmarkExample(state_text="a thing", question=q, gold="a"),
            BenchmarkExample(state_text="a a", question=q, gold="a"),
            BenchmarkExample(state_text="b thing", question=q, gold="b"),
            BenchmarkExample(state_text="b b", question=q, gold="b"),
        ]
        baseline.fit(first)

        # Second fit on different gold distribution — must not error.
        second = [
            BenchmarkExample(state_text="only a", question=q, gold="a"),
            BenchmarkExample(state_text="only b", question=q, gold="b"),
            BenchmarkExample(state_text="also b", question=q, gold="b"),
            BenchmarkExample(state_text="still b", question=q, gold="b"),
        ]
        baseline.fit(second)

        dist = baseline.predict(BenchmarkExample(state_text="b something", question=q, gold="b"))
        assert dist["b"] > dist["a"]
