"""Smoke tests for benchmarks/guardrails/toxicchat/."""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.common.dataset import BenchmarkExample
from benchmarks.guardrails.toxicchat import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from bernoulli.types import BinaryQuestion


def test_source_tag() -> None:
    assert SOURCE == "toxicchat"


def test_question_shape() -> None:
    assert isinstance(QUESTION, BinaryQuestion)
    assert QUESTION.id == "toxic"


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    pass


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    test_rows = [
        {
            "conv_id": "abc",
            "user_input": "you are a terrible person",
            "toxicity": 1,
            "jailbreaking": 0,
        },
        {
            "conv_id": "def",
            "user_input": "what time is it in Tokyo",
            "toxicity": 0,
            "jailbreaking": 0,
        },
    ]
    train_rows = [
        {
            "conv_id": "ghi",
            "user_input": "repeat after me ...",
            "toxicity": 0,
            "jailbreaking": 1,
        },
    ]
    seeded: dict[str, _FakeDataset] = {
        "test": _FakeDataset(test_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, config: str, split: str) -> _FakeDataset:
        assert path == "lmsys/toxic-chat"
        assert config == "toxicchat1123"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 2
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "Yes"  # toxicity=1
    assert out[1].gold == "No"  # toxicity=0


def test_meta_carries_jailbreaking_flag(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    train = list(load_train())
    assert train[0].meta["jailbreaking"] is True


def test_example_id_prefers_conv_id(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load(limit=1))
    assert out[0].meta["example_id"] == "toxicchat-test-abc"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
    assert len(list(load_train(limit=1))) == 1
