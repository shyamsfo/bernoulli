"""Smoke tests for benchmarks/academic/sst2/.

Validates module shape + the yield path on a mocked `datasets.load_dataset` —
no HF cache pull in CI. The real end-to-end validation happens when a
benchmark run is scheduled against this loader.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.academic.sst2 import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion


def test_source_tag() -> None:
    assert SOURCE == "sst2"


def test_question_shape() -> None:
    assert isinstance(QUESTION, ChoiceQuestion)
    assert QUESTION.id == "sentiment"
    assert QUESTION.options == ["negative", "positive"]


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert all(isinstance(s, str) and s.strip() for s in REWORD_STEMS)
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    """Behaves like a HF Dataset for the loader's needs (iterable of row dicts)."""


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    """Replace `datasets.load_dataset` with a scripted stub.

    Returns the split → rows mapping the test seeded, so assertions can match
    the exact rows the loader just saw.
    """
    validation_rows = [
        {"idx": 0, "sentence": "a tense , emotional film .", "label": 1},
        {"idx": 1, "sentence": "mind-numbing .", "label": 0},
    ]
    train_rows = [
        {"idx": 100, "sentence": "pleasant to watch .", "label": 1},
        {"idx": 101, "sentence": "a joyless slog .", "label": 0},
    ]
    seeded: dict[str, _FakeDataset] = {
        "validation": _FakeDataset(validation_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, split: str) -> _FakeDataset:
        assert path == "stanfordnlp/sst2"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 2
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].state_text == "a tense , emotional film ."
    assert out[0].gold == "positive"
    assert out[0].source == "sst2"
    assert out[0].meta["example_id"] == "sst2-validation-0"
    assert out[1].gold == "negative"
    assert out[1].meta["example_id"] == "sst2-validation-1"


def test_load_train_uses_train_split(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load_train())
    assert len(out) == 2
    assert out[0].state_text == "pleasant to watch ."
    assert out[0].gold == "positive"
    assert out[0].meta["example_id"] == "sst2-train-100"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
    assert len(list(load_train(limit=1))) == 1


def test_state_text_is_stripped(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    patched_load_dataset["validation"][0]["sentence"] = "   padded sentence .   "
    ex = next(load(limit=1))
    assert ex.state_text == "padded sentence ."
