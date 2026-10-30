"""Smoke tests for benchmarks/academic/paws/.

First benchmark in the suite with a BinaryQuestion, so tests also assert
Yes/No gold mapping + the sentence-pair state_text format.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.academic.paws import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import BinaryQuestion


def test_source_tag() -> None:
    assert SOURCE == "paws"


def test_question_shape() -> None:
    assert isinstance(QUESTION, BinaryQuestion)
    assert QUESTION.id == "paraphrase"


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert all(isinstance(s, str) and s.strip() for s in REWORD_STEMS)
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    pass


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    test_rows = [
        {
            "id": 1,
            "sentence1": "The cat sat on the mat.",
            "sentence2": "On the mat the cat sat.",
            "label": 1,
        },
        {
            "id": 2,
            "sentence1": "He went to the store.",
            "sentence2": "She went to the park.",
            "label": 0,
        },
    ]
    train_rows = [
        {
            "id": 100,
            "sentence1": "Dogs bark at strangers.",
            "sentence2": "Strangers are barked at by dogs.",
            "label": 1,
        },
    ]
    seeded: dict[str, _FakeDataset] = {
        "test": _FakeDataset(test_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, config: str, split: str) -> _FakeDataset:
        assert path == "google-research-datasets/paws"
        assert config == "labeled_final"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 2
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "Yes"  # label 1 → paraphrase
    assert out[1].gold == "No"  # label 0 → not a paraphrase
    assert out[0].source == "paws"


def test_state_text_formats_both_sentences(
    patched_load_dataset: dict[str, _FakeDataset],
) -> None:
    out = list(load())
    assert out[0].state_text == (
        "Sentence 1: The cat sat on the mat.\nSentence 2: On the mat the cat sat."
    )


def test_example_id_uses_dataset_id_when_available(
    patched_load_dataset: dict[str, _FakeDataset],
) -> None:
    out = list(load())
    # Dataset ships an "id" column — the loader prefers it over the row index.
    assert out[0].meta["example_id"] == "paws-test-1"
    assert out[1].meta["example_id"] == "paws-test-2"


def test_load_train_uses_train_split(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load_train())
    assert len(out) == 1
    assert out[0].gold == "Yes"
    assert out[0].meta["example_id"] == "paws-train-100"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
