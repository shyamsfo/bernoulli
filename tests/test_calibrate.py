"""Calibrator unit tests. Pure numpy + scipy, no model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bernoulli.calibrate import (
    Calibration,
    apply_temperature,
    fit_temperature,
    load_calibration,
    new_version,
    save_calibration,
)


def _overconfident(gold: list[str], n: int) -> list[dict[str, float]]:
    """Return `n` predictions where the model always says 0.99 confidence,
    but is only right half the time. Perfect training set for pushing T > 1."""
    preds = []
    for i, g in enumerate(gold):
        # Alternate: half put all mass on correct, half on wrong — all at 0.99 confidence
        if i % 2 == 0:
            preds.append({"pos": 0.99 if g == "pos" else 0.01, "neg": 0.99 if g == "neg" else 0.01})
        else:
            preds.append({"pos": 0.99 if g == "neg" else 0.01, "neg": 0.99 if g == "pos" else 0.01})
    return preds


class TestFitTemperature:
    def test_perfectly_calibrated_gives_t_near_one(self) -> None:
        """When model says 70% and is right 70% of the time, T should be ~1."""
        gold = ["pos"] * 7 + ["neg"] * 3
        preds = [{"pos": 0.7, "neg": 0.3}] * 10
        t = fit_temperature(gold, preds)
        assert 0.8 < t < 1.4

    def test_overconfident_model_gets_t_greater_than_one(self) -> None:
        """Model always says 0.99 confidence but is only 50% accurate.
        Correct calibration pulls T much higher (cooling the model)."""
        gold = ["pos"] * 50 + ["neg"] * 50
        preds = _overconfident(gold, 100)
        t = fit_temperature(gold, preds)
        assert t > 1.5

    def test_empty_input_returns_one(self) -> None:
        assert fit_temperature([], []) == 1.0

    def test_rejects_gold_outside_label_set(self) -> None:
        gold = ["z"]
        preds = [{"a": 0.5, "b": 0.5}]
        with pytest.raises(ValueError, match="not in label set"):
            fit_temperature(gold, preds)


class TestApplyTemperature:
    def test_t_equal_one_is_identity_up_to_norm(self) -> None:
        probs = {"a": 0.7, "b": 0.3}
        out = apply_temperature(probs, 1.0)
        # T=1 passes probs through unchanged (modulo numerical noise)
        assert out["a"] == pytest.approx(0.7, abs=1e-5)
        assert out["b"] == pytest.approx(0.3, abs=1e-5)

    def test_high_t_moves_toward_uniform(self) -> None:
        probs = {"a": 0.9, "b": 0.1}
        out = apply_temperature(probs, 5.0)
        # Cooling spreads mass toward uniform (0.5, 0.5)
        assert out["a"] < 0.9
        assert out["b"] > 0.1

    def test_low_t_sharpens(self) -> None:
        probs = {"a": 0.6, "b": 0.4}
        out = apply_temperature(probs, 0.3)
        # Sharpening pushes the winner further toward 1
        assert out["a"] > 0.6
        assert out["b"] < 0.4

    def test_preserves_sum_to_one(self) -> None:
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        for t in [0.3, 1.0, 2.5]:
            out = apply_temperature(probs, t)
            assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)


class TestCalibrationRoundtrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cal = Calibration(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            revision="abc123",
            version="2026-10-05",
            temperatures={"choice": 1.3, "binary": 0.9, "rating": 1.1},
        )
        path = tmp_path / "calibration" / "qwen.json"
        save_calibration(path, cal)
        loaded = load_calibration(path)
        assert loaded.model == cal.model
        assert loaded.revision == cal.revision
        assert loaded.version == cal.version
        assert loaded.temperatures == cal.temperatures

    def test_t_for_defaults_to_one(self) -> None:
        cal = Calibration(model="x", revision=None, version="2026-10-05", temperatures={})
        assert cal.t_for("choice") == 1.0

    def test_t_for_known_type(self) -> None:
        cal = Calibration(
            model="x", revision=None, version="2026-10-05", temperatures={"choice": 1.5}
        )
        assert cal.t_for("choice") == 1.5

    def test_new_version_is_iso_date(self) -> None:
        v = new_version()
        assert len(v) == 10 and v[4] == "-" and v[7] == "-"


class TestFileFormat:
    def test_json_shape(self, tmp_path: Path) -> None:
        cal = Calibration(
            model="m", revision="r", version="2026-10-05", temperatures={"choice": 1.2}
        )
        path = tmp_path / "cal.json"
        save_calibration(path, cal)
        data = json.loads(path.read_text())
        assert set(data.keys()) == {"model", "revision", "version", "temperatures"}
        assert isinstance(data["temperatures"], dict)


class TestFitImprovesNLL:
    def test_t_from_overconfident_improves_nll(self) -> None:
        """Sanity check: the fitted T should reduce NLL on the training set."""
        from evals.metrics import nll

        gold = ["pos"] * 50 + ["neg"] * 50
        preds = _overconfident(gold, 100)
        nll_before = nll(gold, preds)
        t = fit_temperature(gold, preds)
        calibrated = [apply_temperature(p, t) for p in preds]
        nll_after = nll(gold, calibrated)
        assert nll_after < nll_before
        # And ECE should also drop — overconfidence is the whole problem
        from evals.metrics import ece

        assert ece(gold, calibrated) < ece(gold, preds)
