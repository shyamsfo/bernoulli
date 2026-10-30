"""Pytest fixtures.

The HFScorer fixture is session-scoped so GPU tests reuse the loaded model
(load is ~2 min, each forward pass is ~0.5s).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.gpu tests when no CUDA device is visible."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def hf_tokenizer() -> object:
    """Load just the tokenizer once (fast, ~1s). No GPU required.

    Tests use the dev-tier 7B tokenizer by default — the production AWQ model
    is slower to download and its tokenizer is tokenization-equivalent for the
    label-token checks that matter here. Set BERNOULLI_TOKENIZER_MODEL_ID to
    swap in a different one.
    """
    from transformers import AutoTokenizer

    model_id = os.environ.get("BERNOULLI_TOKENIZER_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    revision = os.environ.get(
        "BERNOULLI_TOKENIZER_REVISION", "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    )
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


@pytest.fixture(scope="session")
def hf_scorer() -> Iterator[object]:
    """Load the full backbone once per test session (gpu tests only)."""
    from bernoulli.config import load_settings
    from bernoulli.scorer import HFScorer

    settings = load_settings()
    scorer = HFScorer(
        settings.model_id,
        revision=settings.model_revision,
        dtype=settings.dtype,
        device=settings.device,
    )
    yield scorer
