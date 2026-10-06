"""Unit tests for the eval metrics. Pure numpy, no model, no GPU."""

from __future__ import annotations

import math

import pytest

from benchmarks.common.metrics import (
    accuracy,
    brier,
    coverage_curve,
    ece,
    latency_summary,
    macro_f1,
    nll,
    stability,
)


class TestAccuracy:
    def test_all_correct(self) -> None:
        gold = ["a", "b", "a"]
        preds = [
            {"a": 0.9, "b": 0.1},
            {"a": 0.1, "b": 0.9},
            {"a": 0.8, "b": 0.2},
        ]
        assert accuracy(gold, preds) == 1.0

    def test_half_correct(self) -> None:
        gold = ["a", "b"]
        preds = [{"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.1}]
        assert accuracy(gold, preds) == 0.5

    def test_empty(self) -> None:
        assert accuracy([], []) == 0.0


class TestMacroF1:
    def test_perfect_two_class(self) -> None:
        gold = ["a", "a", "b", "b"]
        preds = [
            {"a": 0.9, "b": 0.1},
            {"a": 0.9, "b": 0.1},
            {"a": 0.1, "b": 0.9},
            {"a": 0.1, "b": 0.9},
        ]
        assert macro_f1(gold, preds) == 1.0

    def test_balanced_class_imbalance_vs_micro(self) -> None:
        """macro_f1 should be lower than micro accuracy when the model
        ignores the minority class."""
        gold = ["a"] * 9 + ["b"]
        preds = [{"a": 0.9, "b": 0.1}] * 10  # always predicts a
        assert accuracy(gold, preds) == 0.9  # micro-accuracy is 0.9
        # macro F1: a_f1 = 2*9/(2*9 + 1 + 0) = 18/19 ≈ 0.947; b_f1 = 0
        # macro avg ≈ 0.474
        assert macro_f1(gold, preds) < 0.5


class TestECE:
    def test_perfectly_calibrated(self) -> None:
        """A model that assigns P=0.7 and is right 70% of the time has ECE=0."""
        gold = ["a"] * 7 + ["b"] * 3
        preds = [{"a": 0.7, "b": 0.3}] * 10
        assert ece(gold, preds) < 0.01

    def test_overconfident(self) -> None:
        """P=0.99 but only right 50% of the time → ECE ≈ 0.49."""
        gold = ["a"] * 5 + ["b"] * 5
        preds = [{"a": 0.99, "b": 0.01}] * 10
        assert ece(gold, preds) > 0.4


class TestBrier:
    def test_perfect(self) -> None:
        gold = ["a", "b"]
        preds = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
        assert brier(gold, preds) == 0.0

    def test_uniform(self) -> None:
        gold = ["a", "b"]
        preds = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
        # (0.5)^2 + (0.5)^2 = 0.5 per example
        assert math.isclose(brier(gold, preds), 0.5, abs_tol=1e-6)


class TestNLL:
    def test_perfect(self) -> None:
        gold = ["a"]
        preds = [{"a": 1.0, "b": 0.0}]
        assert nll(gold, preds) < 1e-6

    def test_worst(self) -> None:
        gold = ["a"]
        preds = [{"a": 0.0, "b": 1.0}]
        # NLL is -log(eps), clipped to a huge finite value
        assert nll(gold, preds) > 10


class TestAlignmentErrors:
    def test_rejects_mismatched_keys(self) -> None:
        gold = ["a", "b"]
        preds = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "c": 0.5}]
        with pytest.raises(ValueError, match="keys"):
            accuracy(gold, preds)

    def test_rejects_gold_outside_label_set(self) -> None:
        gold = ["a", "z"]
        preds = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
        with pytest.raises(ValueError, match="not in label set"):
            brier(gold, preds)


class TestLatencySummary:
    def test_percentiles(self) -> None:
        s = latency_summary(list(range(1, 101)))  # 1..100
        assert s["p50_ms"] == pytest.approx(50.5)
        assert s["p95_ms"] == pytest.approx(95.05)
        assert s["mean_ms"] == pytest.approx(50.5)

    def test_empty(self) -> None:
        assert latency_summary([]) == {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}


class TestCoverageCurve:
    def test_perfect_classifier_is_perfect_at_every_coverage(self) -> None:
        gold = ["a", "b", "a", "b"]
        preds = [
            {"a": 0.95, "b": 0.05},
            {"a": 0.05, "b": 0.95},
            {"a": 0.80, "b": 0.20},
            {"a": 0.20, "b": 0.80},
        ]
        cov = coverage_curve(gold, preds)
        assert cov[0.95] == 1.0
        assert cov[0.50] == 1.0

    def test_kept_subset_is_the_most_confident(self) -> None:
        gold = ["a", "b", "a", "b"]
        # The two most confident predictions (0.99 each) are both wrong — on purpose.
        # At coverage=0.5 we keep only those two: accuracy should be 0.0.
        # At full coverage we keep all four — the two unconfident ones are correct — accuracy = 0.5.
        preds = [
            {"a": 0.01, "b": 0.99},  # most confident, wrong
            {"a": 0.99, "b": 0.01},  # most confident, wrong
            {"a": 0.60, "b": 0.40},  # correct
            {"a": 0.40, "b": 0.60},  # correct
        ]
        cov = coverage_curve(gold, preds, levels=[1.0, 0.5])
        assert cov[1.0] == 0.5
        assert cov[0.5] == 0.0

    def test_rejects_out_of_range_level(self) -> None:
        with pytest.raises(ValueError, match="coverage level"):
            coverage_curve(["a"], [{"a": 1.0, "b": 0.0}], levels=[1.5])
        with pytest.raises(ValueError, match="coverage level"):
            coverage_curve(["a"], [{"a": 1.0, "b": 0.0}], levels=[0.0])

    def test_empty(self) -> None:
        cov = coverage_curve([], [], levels=[0.95, 0.5])
        assert cov == {0.95: 0.0, 0.5: 0.0}


class TestStability:
    def test_all_variants_agree_with_reference(self) -> None:
        ref = ["a", "b", "a"]
        variants = [["a", "b", "a"], ["a", "b", "a"]]
        assert stability(ref, variants) == 1.0

    def test_one_example_flips_under_one_variant(self) -> None:
        ref = ["a", "b", "a", "b"]
        variants = [
            ["a", "b", "a", "b"],  # all agree
            ["a", "b", "x", "b"],  # third example flips
        ]
        # 3 of 4 examples remain stable across *all* variants
        assert stability(ref, variants) == 0.75

    def test_no_variants_counts_as_fully_stable(self) -> None:
        """With zero variants, there's nothing to disagree — treat as stable."""
        assert stability(["a", "b"], []) == 1.0

    def test_empty_reference(self) -> None:
        assert stability([], [["a"]]) == 0.0

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="length"):
            stability(["a", "b"], [["a"]])
