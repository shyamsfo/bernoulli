"""Smoke tests for benchmarks/academic/tweeteval_emotion/.

Monkeypatched datasets.load_dataset — no HF cache pull in CI. Deviation
from the earlier benchmark tests: TweetEval loads with a config
argument (`"emotion"`), so the fake load_dataset signature here checks
for the third positional arg too.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.academic.tweeteval_emotion import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion


def test_source_tag() -> None:
    assert SOURCE == "tweeteval_emotion"


def test_question_shape() -> None:
    assert isinstance(QUESTION, ChoiceQuestion)
    assert QUESTION.id == "emotion"
    assert QUESTION.options == ["anger", "joy", "optimism", "sadness"]


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert all(isinstance(s, str) and s.strip() for s in REWORD_STEMS)
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    pass


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    test_rows = [
        {"text": "so excited for the game tonight", "label": 1},
        {"text": "this is infuriating", "label": 0},
    ]
    train_rows = [
        {"text": "feeling hopeful about the week", "label": 2},
        {"text": "down and out today", "label": 3},
    ]
    seeded: dict[str, _FakeDataset] = {
        "test": _FakeDataset(test_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, config: str, split: str) -> _FakeDataset:
        assert path == "cardiffnlp/tweet_eval"
        assert config == "emotion"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 2
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "joy"
    assert out[0].source == "tweeteval_emotion"
    assert out[0].meta["example_id"] == "tweeteval_emotion-test-0"
    assert out[1].gold == "anger"


def test_load_train_uses_train_split(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load_train())
    assert len(out) == 2
    assert out[0].gold == "optimism"
    assert out[0].meta["example_id"] == "tweeteval_emotion-train-0"
    assert out[1].gold == "sadness"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
    assert len(list(load_train(limit=1))) == 1


def test_state_text_is_stripped(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    patched_load_dataset["test"][0]["text"] = "   padded tweet   "
    ex = next(load(limit=1))
    assert ex.state_text == "padded tweet"
