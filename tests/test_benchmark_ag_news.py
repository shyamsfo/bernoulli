"""Smoke tests for benchmarks/academic/ag_news/.

Same shape as test_benchmark_sst2.py: monkeypatched datasets.load_dataset,
no HF cache pull in CI. End-to-end validation happens at benchmark-run time.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.academic.ag_news import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion


def test_source_tag() -> None:
    assert SOURCE == "ag_news"


def test_question_shape() -> None:
    assert isinstance(QUESTION, ChoiceQuestion)
    assert QUESTION.id == "topic"
    assert QUESTION.options == ["World", "Sports", "Business", "Sci/Tech"]


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert all(isinstance(s, str) and s.strip() for s in REWORD_STEMS)
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    pass


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    test_rows = [
        {"text": "Stocks slip on energy fears .", "label": 2},
        {"text": "Rockets edge Mavericks in overtime .", "label": 1},
    ]
    train_rows = [
        {"text": "Prime minister visits capital .", "label": 0},
        {"text": "New GPU launched by chipmaker .", "label": 3},
    ]
    seeded: dict[str, _FakeDataset] = {
        "test": _FakeDataset(test_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, split: str) -> _FakeDataset:
        assert path == "fancyzhx/ag_news"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 2
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "Business"
    assert out[0].source == "ag_news"
    assert out[0].meta["example_id"] == "ag_news-test-0"
    assert out[1].gold == "Sports"


def test_load_train_uses_train_split(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load_train())
    assert len(out) == 2
    assert out[0].gold == "World"
    assert out[0].meta["example_id"] == "ag_news-train-0"
    assert out[1].gold == "Sci/Tech"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
    assert len(list(load_train(limit=1))) == 1


def test_state_text_is_stripped(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    patched_load_dataset["test"][0]["text"] = "   padded article .   "
    ex = next(load(limit=1))
    assert ex.state_text == "padded article ."
