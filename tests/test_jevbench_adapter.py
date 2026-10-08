"""Smoke tests for the JevBench adapter.

No network, no running Bernoulli server. We stub BernoulliHTTP so the
unit under test is the schema translation layer:

- JevBench task → BenchmarkExample (per question kind).
- Server distribution → JevBench canonical-label order (per question kind).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The adapter module auto-adds upstream/ to sys.path; make sure that happens
# before we import anything else from the upstream package.
_UPSTREAM = Path(__file__).parent.parent / "benchmarks" / "jevbench" / "upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from benchmarks.jevbench.adapter import BernoulliLocalAdapter  # noqa: E402
from bernoulli.types import BinaryQuestion, ChoiceQuestion, RatingQuestion  # noqa: E402


def _task_noul(state: str = "a policy state", instructions: str = "Permitted?") -> SimpleNamespace:
    return SimpleNamespace(
        id="t-noul",
        state=state,
        question={
            "type": "noul",
            "instructions": instructions,
            "criteria": {"true": "all conditions met", "false": "something is missing"},
        },
        labels=["no", "yes"],
    )


def _task_choice() -> SimpleNamespace:
    return SimpleNamespace(
        id="t-choice",
        state="customer wrote in angry about a charge",
        question={
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {
                "refund": "wants money back",
                "info": "wants an explanation",
                "other": "unclear",
            },
        },
        labels=["refund", "info", "other"],
    )


def _task_score() -> SimpleNamespace:
    return SimpleNamespace(
        id="t-score",
        state="a middling review",
        question={
            "type": "score",
            "instructions": "Rate severity",
            "criteria": ["minor", "moderate", "major"],  # len=3 → levels 0,1,2
        },
        labels=["0", "1", "2"],
    )


def test_adapter_class_shape_matches_contract() -> None:
    """Match the informal class shape every file in jevbench/adapters/ follows."""
    adapter = BernoulliLocalAdapter(endpoint="http://127.0.0.1:8000", model="bernoulli-qwen-7b")
    assert adapter.name == "bernoulli_local"
    assert adapter.cost_basis == "local_self_hosted_no_api_tariff"
    assert hasattr(adapter, "run")


class TestBuildExample:
    def test_noul_builds_binary_question(self) -> None:
        a = BernoulliLocalAdapter()
        ex, flavor = a._build_example(_task_noul(state="s", instructions="go/no-go?"))
        assert flavor == "noul"
        assert isinstance(ex.question, BinaryQuestion)
        assert ex.question.id == "decision"
        assert ex.question.prompt == "go/no-go?"
        assert ex.state_text == "s"

    def test_choice_builds_choice_question_with_rubric_in_prompt(self) -> None:
        a = BernoulliLocalAdapter()
        ex, flavor = a._build_example(_task_choice())
        assert flavor == "choice"
        assert isinstance(ex.question, ChoiceQuestion)
        assert ex.question.options == ["refund", "info", "other"]
        assert "refund: wants money back" in ex.question.prompt
        assert "info: wants an explanation" in ex.question.prompt

    def test_score_builds_rating_question_shifted_to_one_based(self) -> None:
        """RatingQuestion scale is 1-based; _shape_probs shifts back to 0-based."""
        a = BernoulliLocalAdapter()
        ex, flavor = a._build_example(_task_score())
        assert flavor == "score"
        assert isinstance(ex.question, RatingQuestion)
        assert ex.question.scale == (1, 3)
        assert "0: minor" in ex.question.prompt
        assert "2: major" in ex.question.prompt

    def test_non_string_state_json_serialised(self) -> None:
        a = BernoulliLocalAdapter()
        task = _task_noul()
        task.state = {"ticket": "abc", "body": "need help"}
        ex, _ = a._build_example(task)
        assert "ticket" in ex.state_text and "abc" in ex.state_text

    def test_unknown_type_raises(self) -> None:
        a = BernoulliLocalAdapter()
        bad = _task_noul()
        bad.question["type"] = "bogus"
        with pytest.raises(ValueError, match="unknown JevBench question type"):
            a._build_example(bad)


class TestShapeProbs:
    def test_noul_lowercases_and_reorders(self) -> None:
        a = BernoulliLocalAdapter()
        # Server returns capitalised Yes/No; JevBench expects ["no", "yes"].
        shaped = a._shape_probs({"Yes": 0.72, "No": 0.28}, "noul", _task_noul())
        assert shaped == {"no": pytest.approx(0.28), "yes": pytest.approx(0.72)}

    def test_choice_preserves_canonical_order(self) -> None:
        a = BernoulliLocalAdapter()
        shaped = a._shape_probs(
            {"other": 0.1, "refund": 0.7, "info": 0.2}, "choice", _task_choice()
        )
        # Insertion-ordered dict; comparing by (key, value) via list(items())
        # catches both value and order.
        assert list(shaped.items()) == [("refund", 0.7), ("info", 0.2), ("other", 0.1)]

    def test_score_shifts_one_based_to_zero_based(self) -> None:
        a = BernoulliLocalAdapter()
        # Server returned 1-based keys ("1".."3"); shaped should be "0".."2".
        shaped = a._shape_probs({"1": 0.2, "2": 0.5, "3": 0.3}, "score", _task_score())
        assert shaped == {"0": pytest.approx(0.2), "1": pytest.approx(0.5), "2": pytest.approx(0.3)}


class _MockBernoulliHTTP:
    """Captures the call + returns a scripted distribution."""

    def __init__(self, canned: dict[str, float]) -> None:
        self._canned = canned
        self.called_with: object = None

    def predict(self, example):
        self.called_with = example
        return self._canned


class TestRunHappyPath:
    def test_run_returns_native_probs_on_noul(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: run(task) → DecisionResult with ok=True, probs reshaped."""
        mock = _MockBernoulliHTTP({"Yes": 0.8, "No": 0.2})
        monkeypatch.setattr("benchmarks.jevbench.adapter.BernoulliHTTP", lambda **kw: mock)
        a = BernoulliLocalAdapter()
        res = a.run(_task_noul())
        assert res.ok is True
        assert res.probs == {"no": pytest.approx(0.2), "yes": pytest.approx(0.8)}
        assert res.probs_source == "native"
        assert res.model == "bernoulli-qwen-7b"
        assert res.adapter == "bernoulli_local"
        assert res.latency_s >= 0.0
        assert res.error is None

    def test_run_captures_exception_without_crashing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Boom:
            def predict(self, _ex):
                raise RuntimeError("network down")

        monkeypatch.setattr("benchmarks.jevbench.adapter.BernoulliHTTP", lambda **kw: _Boom())
        a = BernoulliLocalAdapter()
        res = a.run(_task_noul())
        assert res.ok is False
        assert res.probs is None
        assert "network down" in res.error
