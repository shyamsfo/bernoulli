"""Smoke tests for benchmarks/academic/banking77/.

Same shape as test_benchmark_sst2.py — monkeypatched datasets.load_dataset,
no HF cache pull in CI. Deviation from the sst2/ag_news tests: Banking77
builds its ChoiceQuestion dynamically from the dataset, so the test also
asserts that the derived option list is sorted by integer label (not
iteration order).
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.academic.banking77 import (
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
    question,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion


def test_source_tag() -> None:
    assert SOURCE == "banking77"


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert all(isinstance(s, str) and s.strip() for s in REWORD_STEMS)
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeDataset(list[dict[str, Any]]):
    pass


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    # Intentionally shuffled label order in the rows — asserts the loader
    # sorts options by int label, not by iteration order.
    test_rows = [
        {"text": "where's my refund", "label": 2, "label_text": "refund_not_showing_up"},
        {"text": "can I use apple pay", "label": 0, "label_text": "apple_pay_or_google_pay"},
        {"text": "activate my new card", "label": 1, "label_text": "activate_my_card"},
    ]
    train_rows = [
        {"text": "where is my card", "label": 1, "label_text": "activate_my_card"},
        {"text": "refund is late", "label": 2, "label_text": "refund_not_showing_up"},
    ]
    seeded: dict[str, _FakeDataset] = {
        "test": _FakeDataset(test_rows),
        "train": _FakeDataset(train_rows),
    }

    def fake_load_dataset(path: str, split: str) -> _FakeDataset:
        assert path == "mteb/banking77"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_question_options_sorted_by_int_label(
    patched_load_dataset: dict[str, _FakeDataset],
) -> None:
    q = question()
    assert isinstance(q, ChoiceQuestion)
    assert q.id == "intent"
    # Labels 0, 1, 2 in the fixture → options in that order:
    assert q.options == [
        "apple_pay_or_google_pay",
        "activate_my_card",
        "refund_not_showing_up",
    ]


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 3
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "refund_not_showing_up"
    assert out[0].source == "banking77"
    assert out[0].meta["example_id"] == "banking77-test-0"
    # The example's question is the full 3-option set shared across all examples:
    assert out[0].question.options == out[1].question.options


def test_load_train_uses_train_split(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load_train())
    assert len(out) == 2
    assert out[0].gold == "activate_my_card"
    assert out[0].meta["example_id"] == "banking77-train-0"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1
    assert len(list(load_train(limit=1))) == 1


def test_state_text_is_stripped(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    patched_load_dataset["test"][0]["text"] = "   padded message   "
    ex = next(load(limit=1))
    assert ex.state_text == "padded message"
