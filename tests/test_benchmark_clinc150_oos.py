"""Smoke tests for benchmarks/triage/clinc150_oos/.

Covers the OOS-id-by-name resolution + the Yes/No gold mapping. The
mock dataset needs to expose a `features` attribute with int2str and a
`names` list so the loader's reflection works.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.common.dataset import BenchmarkExample
from benchmarks.triage.clinc150_oos import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from bernoulli.types import BinaryQuestion


def test_source_tag() -> None:
    assert SOURCE == "clinc150_oos"


def test_question_shape() -> None:
    assert isinstance(QUESTION, BinaryQuestion)
    assert QUESTION.id == "in_scope"


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


class _FakeIntentFeature:
    """Just enough of HF datasets' ClassLabel to satisfy the loader."""

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def int2str(self, idx: int) -> str:
        return self.names[idx]


class _FakeFeatures(dict[str, Any]):
    def __init__(self, intent_names: list[str]) -> None:
        super().__init__()
        self["intent"] = _FakeIntentFeature(intent_names)


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]], intent_names: list[str]) -> None:
        self._rows = rows
        self.features = _FakeFeatures(intent_names)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)


@pytest.fixture
def patched_load_dataset(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeDataset]:
    # intent idx: 0 = balance, 1 = travel, 2 = oos
    intent_names = ["balance", "travel", "oos"]

    test_rows = [
        {"text": "what's my checking balance", "intent": 0},  # in-scope
        {"text": "flight from SFO to NYC", "intent": 1},  # in-scope
        {"text": "write me a sonnet", "intent": 2},  # OOS
    ]
    train_rows = [
        {"text": "my account balance", "intent": 0},
        {"text": "what time is it", "intent": 2},
    ]
    seeded = {
        "test": _FakeDataset(test_rows, intent_names),
        "train": _FakeDataset(train_rows, intent_names),
    }

    def fake_load_dataset(path: str, config: str, split: str) -> _FakeDataset:
        assert path == "clinc_oos"
        assert config == "plus"
        return seeded[split]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seeded


def test_load_yields_benchmark_examples(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert len(out) == 3
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].gold == "Yes"  # in-scope
    assert out[1].gold == "Yes"
    assert out[2].gold == "No"  # OOS


def test_meta_carries_intent_fields(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load())
    assert out[0].meta["intent_name"] == "balance"
    assert out[0].meta["intent_id"] == 0
    assert out[0].meta["is_oos"] is False
    assert out[2].meta["intent_name"] == "oos"
    assert out[2].meta["is_oos"] is True


def test_load_train_uses_train_split(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    out = list(load_train())
    assert len(out) == 2
    assert out[1].gold == "No"


def test_limit_caps_output(patched_load_dataset: dict[str, _FakeDataset]) -> None:
    assert len(list(load(limit=1))) == 1


def test_missing_oos_label_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _FakeDataset(rows=[], intent_names=["balance", "travel"])  # no "oos"

    def fake_load_dataset(path: str, config: str, split: str) -> _FakeDataset:
        return bad

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    with pytest.raises(RuntimeError, match="expected an intent named 'oos'"):
        list(load())
