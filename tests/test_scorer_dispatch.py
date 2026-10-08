"""Unit tests for the config-driven backbone dispatch in bernoulli/scorer.py.

The actual model-load path is exercised by the GPU-only session fixture in
`tests/conftest.py:hf_scorer` (which only runs on bernoulli where a GPU is
available). These tests are the CPU-only counterpart that covers the
VL-vs-text dispatch logic without pulling any weights.
"""

from __future__ import annotations

from types import SimpleNamespace

from bernoulli.scorer import _is_vl_config


class TestIsVLConfig:
    def test_vl_config_detected_by_non_none_vision_config(self) -> None:
        """Qwen2.5-VL + friends carry a populated `vision_config` sub-config."""
        config = SimpleNamespace(
            model_type="qwen2_5_vl",
            vision_config=SimpleNamespace(num_hidden_layers=32),
        )
        assert _is_vl_config(config) is True

    def test_text_config_without_vision_config_attr(self) -> None:
        """Classical decoder-only LMs don't have the attribute at all."""
        config = SimpleNamespace(model_type="llama")
        assert _is_vl_config(config) is False

    def test_text_config_with_explicit_none_vision_config(self) -> None:
        """Some configs carry `vision_config=None` as a placeholder — treat as text-only."""
        config = SimpleNamespace(model_type="qwen2", vision_config=None)
        assert _is_vl_config(config) is False
