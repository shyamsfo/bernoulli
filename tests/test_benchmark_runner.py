"""Unit tests for benchmarks/run.py.

Covers the orchestration bits — baseline resolution, trainable-fit path,
metric aggregation, report writing. All with mock baselines; no HF, no
network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common.baselines import Distribution
from benchmarks.common.dataset import BenchmarkExample
from benchmarks.run import (
    _resolve_benchmark_module,
    _run_one_baseline,
    write_report,
)
from bernoulli.types import ChoiceQuestion


class _AlwaysPositive:
    """Mock baseline whose top-1 is 'positive' with varying confidence."""

    name = "always-positive"

    def __init__(self, confidence: float = 0.9) -> None:
        self._conf = confidence

    def predict(self, example: BenchmarkExample) -> Distribution:
        return {"negative": 1.0 - self._conf, "positive": self._conf}


class _TrainableBaseline:
    """Trainable mock — tracks whether fit() was called before predict()."""

    name = "trainable"

    def __init__(self) -> None:
        self.fitted = False
        self.fit_count = 0

    def fit(self, train_examples: list[BenchmarkExample]) -> None:
        self.fitted = True
        self.fit_count = len(train_examples)

    def predict(self, example: BenchmarkExample) -> Distribution:
        if not self.fitted:
            raise RuntimeError("predict before fit")
        return {"negative": 0.5, "positive": 0.5}


def _choice_example(state: str, gold: str) -> BenchmarkExample:
    q = ChoiceQuestion(id="x", prompt="pos or neg?", options=["negative", "positive"])
    return BenchmarkExample(state_text=state, question=q, gold=gold, source="t")


def test_resolve_benchmark_module_imports_academic_sst2() -> None:
    module = _resolve_benchmark_module("academic/sst2")
    assert hasattr(module, "load")
    assert hasattr(module, "load_train")


class TestRunOneBaseline:
    def test_computes_expected_metrics_on_mock(self) -> None:
        examples = [
            _choice_example("great", gold="positive"),
            _choice_example("awesome", gold="positive"),
            _choice_example("meh", gold="negative"),
        ]
        result = _run_one_baseline(
            "always-positive",
            _AlwaysPositive(confidence=0.9),
            examples,
            reword_stems=(),
            train_examples=None,
            train_limit=None,
        )
        assert result.name == "always-positive"
        # 2 of 3 correct = 0.6667
        assert result.accuracy == pytest.approx(2 / 3, rel=1e-3)
        assert result.n_examples == 3
        assert result.stability_reword == 1.0  # no rewords ⇒ fully stable

    def test_fits_trainable_before_predict(self) -> None:
        baseline = _TrainableBaseline()
        eval_examples = [_choice_example("s1", gold="positive")]
        train_examples = [
            _choice_example("t1", gold="positive"),
            _choice_example("t2", gold="negative"),
            _choice_example("t3", gold="positive"),
        ]
        _run_one_baseline(
            "trainable",
            baseline,
            eval_examples,
            reword_stems=(),
            train_examples=train_examples,
            train_limit=None,
        )
        assert baseline.fitted
        assert baseline.fit_count == 3

    def test_train_limit_caps_training_set(self) -> None:
        baseline = _TrainableBaseline()
        train_examples = [_choice_example(f"t{i}", gold="positive") for i in range(10)]
        _run_one_baseline(
            "trainable",
            baseline,
            [_choice_example("eval", gold="positive")],
            reword_stems=(),
            train_examples=train_examples,
            train_limit=4,
        )
        assert baseline.fit_count == 4

    def test_trainable_without_train_examples_raises(self) -> None:
        baseline = _TrainableBaseline()
        with pytest.raises(RuntimeError, match="needs a training set"):
            _run_one_baseline(
                "trainable",
                baseline,
                [_choice_example("eval", gold="positive")],
                reword_stems=(),
                train_examples=None,
                train_limit=None,
            )

    def test_training_set_summary_in_note(self) -> None:
        baseline = _TrainableBaseline()
        train_examples = [_choice_example(f"t{i}", gold="positive") for i in range(5)]
        result = _run_one_baseline(
            "trainable",
            baseline,
            [_choice_example("eval", gold="positive")],
            reword_stems=(),
            train_examples=train_examples,
            train_limit=None,
        )
        assert "trained on 5" in result.note


class TestWriteReport:
    def test_passes_extra_metrics_fn_through(self) -> None:
        """The runner forwards module.extra_metrics into BaselineMetrics.extras."""
        examples = [_choice_example("great", gold="positive")]

        def fake_extras(
            gold: list[str], preds: list[Distribution], exs: list[BenchmarkExample]
        ) -> dict[str, float]:
            # Return a sentinel value proving the fn ran with the expected shape.
            return {"custom_metric": 0.42}

        result = _run_one_baseline(
            "always-positive",
            _AlwaysPositive(0.8),
            examples,
            reword_stems=(),
            train_examples=None,
            train_limit=None,
            extra_metrics_fn=fake_extras,
        )
        assert result.extras == {"custom_metric": 0.42}

    def test_markdown_and_sidecar_round_trip(self, tmp_path: Path) -> None:
        examples = [
            _choice_example("great", gold="positive"),
            _choice_example("awful", gold="negative"),
        ]
        result = _run_one_baseline(
            "always-positive",
            _AlwaysPositive(0.9),
            examples,
            reword_stems=(),
            train_examples=None,
            train_limit=None,
        )
        out_md = tmp_path / "results" / "2026-10-07.md"
        write_report(
            out_md,
            "academic/sst2",
            [result],
            repro={"benchmark": "academic/sst2", "bernoulli_commit": "abc123"},
        )
        text = out_md.read_text()
        assert "`academic/sst2`" in text
        assert "always-positive" in text
        assert "Reproducibility" in text
        assert "Coverage" in text
        assert "latency" in text.lower()

        sidecar = out_md.with_suffix(".json")
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["benchmark"] == "academic/sst2"
        assert payload["repro"]["bernoulli_commit"] == "abc123"
        assert payload["results"][0]["name"] == "always-positive"
        assert payload["results"][0]["accuracy"] == pytest.approx(0.5)
