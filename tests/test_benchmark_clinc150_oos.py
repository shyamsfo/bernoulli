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

    def shuffle(self, seed: int = 0) -> _FakeDataset:
        # Deterministic, order-preserving stand-in — the real datasets.Dataset
        # permutes the rows, but the mock only needs to return something that
        # iterates. Tests assert on label presence/counts, not ordering.
        return self


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
        assert path == "Clinc/clinc_oos"
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


class TestExtraMetrics:
    def test_oos_auroc_perfect_score_when_scores_separate(self) -> None:
        from benchmarks.triage.clinc150_oos.loader import extra_metrics

        q = QUESTION
        examples = [
            BenchmarkExample(
                state_text="in-scope q",
                question=q,
                gold="Yes",
                source="t",
                meta={"is_oos": False},
            ),
            BenchmarkExample(
                state_text="oos q",
                question=q,
                gold="No",
                source="t",
                meta={"is_oos": True},
            ),
        ]
        # Confidence tracks is_oos perfectly: in-scope → P(No)=0.1, OOS → P(No)=0.9.
        preds = [{"Yes": 0.9, "No": 0.1}, {"Yes": 0.1, "No": 0.9}]
        gold = ["Yes", "No"]
        out = extra_metrics(gold, preds, examples)
        assert out["oos_auroc"] == pytest.approx(1.0)
        assert out["in_scope_accuracy"] == pytest.approx(1.0)
        assert out["oos_recall"] == pytest.approx(1.0)

    def test_oos_recall_counts_argmax_no_only(self) -> None:
        from benchmarks.triage.clinc150_oos.loader import extra_metrics

        q = QUESTION
        # Two OOS examples; baseline labels one "No" (correct recall), the other "Yes" (missed).
        examples = [
            BenchmarkExample(
                state_text="a",
                question=q,
                gold="No",
                source="t",
                meta={"is_oos": True},
            ),
            BenchmarkExample(
                state_text="b",
                question=q,
                gold="No",
                source="t",
                meta={"is_oos": True},
            ),
        ]
        preds = [{"Yes": 0.2, "No": 0.8}, {"Yes": 0.8, "No": 0.2}]
        gold = ["No", "No"]
        out = extra_metrics(gold, preds, examples)
        assert out["oos_recall"] == pytest.approx(0.5)

    def test_degenerate_single_class_returns_safe_defaults(self) -> None:
        from benchmarks.triage.clinc150_oos.loader import extra_metrics

        q = QUESTION
        # All in-scope; no OOS examples.
        examples = [
            BenchmarkExample(
                state_text=f"q{i}",
                question=q,
                gold="Yes",
                source="t",
                meta={"is_oos": False},
            )
            for i in range(3)
        ]
        preds = [{"Yes": 0.9, "No": 0.1} for _ in range(3)]
        gold = ["Yes"] * 3
        out = extra_metrics(gold, preds, examples)
        assert out["oos_auroc"] == 0.5  # degenerate
        assert out["in_scope_accuracy"] == pytest.approx(1.0)
        assert out["oos_recall"] == 0.0
