"""Smoke tests for benchmarks/ratings/yelp_stars/.

First benchmark with RatingQuestion. Covers the 0..4 → "1".."5" label
conversion and the scale carryover.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.common.dataset import BenchmarkExample
from benchmarks.ratings.yelp_stars import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from bernoulli.types import RatingQuestion


def test_source_tag() -> None:
    assert SOURCE == "yelp_stars"


def test_question_shape() -> None:
    assert isinstance(QUESTION, RatingQuestion)
    assert QUESTION.id == "stars"
    assert QUESTION.scale == (1, 5)


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    pass


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    test_rows = [
        {"text": "absolutely terrible, worst ever", "label": 0},  # → "1"
        {"text": "could be better", "label": 2},  # → "3"
        {"text": "best meal of my life", "label": 4},  # → "5"
    ]
    train_rows = [
        {"text": "great service, loved it", "label": 3},  # → "4"
    ]
    seeded: dict[str, _FakeDataset] = {
        "test": _FakeDataset(test_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, split: str) -> _FakeDataset:
        assert path == "yelp_review_full"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_label_shifted_to_one_through_five(
    patched_load_dataset: dict[str, _FakeDataset],
) -> None:
    out = list(load())
    assert len(out) == 3
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "1"
    assert out[1].gold == "3"
    assert out[2].gold == "5"


def test_load_train_matches_convention(
    patched_load_dataset: dict[str, _FakeDataset],
) -> None:
    out = list(load_train())
    assert out[0].gold == "4"
    assert out[0].meta["example_id"] == "yelp-train-0"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
