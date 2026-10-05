"""Config env-override smoke tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bernoulli.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.model_id.startswith("Qwen/")
    assert s.dtype == "bfloat16"
    assert s.scorer == "hf"
    assert s.default_debias == "reverse"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BERNOULLI_MODEL_ID", "test/model")
    monkeypatch.setenv("BERNOULLI_DTYPE", "float16")
    monkeypatch.setenv("BERNOULLI_SCORER", "vllm")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.model_id == "test/model"
    assert s.dtype == "float16"
    assert s.scorer == "vllm"


def test_rejects_invalid_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BERNOULLI_DTYPE", "float8")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_max_len_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BERNOULLI_MAX_MODEL_LEN", "512")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
