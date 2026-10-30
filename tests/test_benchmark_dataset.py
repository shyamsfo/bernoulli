"""Smoke tests for benchmarks.common.dataset.BenchmarkExample.

Keeps the common type honest: field names + defaults match the migration
expectation from evals.example.EvalExample, meta is dict-shaped, and the
dataclass is frozen so examples are safe to pass through async pipelines.
"""

from __future__ import annotations

import dataclasses

import pytest

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import BinaryQuestion, ChoiceQuestion


def test_minimal_construction() -> None:
    q = ChoiceQuestion(id="sentiment", prompt="pos or neg?", options=["negative", "positive"])
    ex = BenchmarkExample(state_text="great movie", question=q, gold="positive")
    assert ex.state_text == "great movie"
    assert ex.gold == "positive"
    assert ex.source == ""
    assert dict(ex.meta) == {}


def test_source_and_meta() -> None:
    q = BinaryQuestion(id="toxic", prompt="toxic?")
    ex = BenchmarkExample(
        state_text="hi",
        question=q,
        gold="No",
        source="wildguard_test",
        meta={"example_id": "wg-0001", "category": "benign"},
    )
    assert ex.source == "wildguard_test"
    assert ex.meta["example_id"] == "wg-0001"
    assert ex.meta["category"] == "benign"


def test_frozen() -> None:
    q = BinaryQuestion(id="x", prompt="?")
    ex = BenchmarkExample(state_text="hi", question=q, gold="Yes")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ex.gold = "No"  # type: ignore[misc]


def test_default_meta_not_shared() -> None:
    """Each BenchmarkExample must get its own meta dict (no mutable-default bug)."""
    q = BinaryQuestion(id="x", prompt="?")
    a = BenchmarkExample(state_text="a", question=q, gold="Yes")
    b = BenchmarkExample(state_text="b", question=q, gold="No")
    assert a.meta is not b.meta
