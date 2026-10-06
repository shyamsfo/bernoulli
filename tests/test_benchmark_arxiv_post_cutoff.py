"""Smoke tests for benchmarks/academic/arxiv_post_cutoff/.

Reads from the committed sample.jsonl rather than monkeypatching anything —
the loader is file-backed, not HF-backed. Tests cover schema, split
filtering, title/abstract state_text format, and the fallback warning
when dataset.jsonl is absent (which is the committed state in 10a).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from benchmarks.academic.arxiv_post_cutoff import (
    BACKBONE_CUTOFF,
    CATEGORIES,
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)
from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion


def test_source_tag() -> None:
    assert SOURCE == "arxiv_post_cutoff"


def test_question_shape() -> None:
    assert isinstance(QUESTION, ChoiceQuestion)
    assert QUESTION.id == "arxiv_category"
    assert QUESTION.options == list(CATEGORIES)


def test_categories_are_distinct() -> None:
    assert len(CATEGORIES) == 4
    assert len(set(CATEGORIES)) == 4


def test_backbone_cutoff_parseable_and_after_qwen25() -> None:
    cutoff = date.fromisoformat(BACKBONE_CUTOFF)
    # Qwen2.5 series: pretraining cutoff is late 2023; our default must be after it.
    assert cutoff >= date(2024, 1, 1)


def test_reword_stems_are_distinct_strings() -> None:
    assert len(REWORD_STEMS) >= 3
    assert all(isinstance(s, str) and s.strip() for s in REWORD_STEMS)
    assert len(set(REWORD_STEMS)) == len(REWORD_STEMS)


_SAMPLE_FILE = (
    Path(__file__).parent.parent
    / "benchmarks"
    / "academic"
    / "arxiv_post_cutoff"
    / "data"
    / "sample.jsonl"
)


def test_sample_file_exists() -> None:
    assert _SAMPLE_FILE.exists(), f"sample dataset must be committed at {_SAMPLE_FILE}"


def test_load_reads_test_split_from_sample() -> None:
    with pytest.warns(UserWarning, match="dataset.jsonl not found"):
        out = list(load())
    assert len(out) == 2  # two test rows in the committed sample
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert out[0].source == SOURCE
    assert {ex.gold for ex in out} == {"cs.CL", "cs.CV"}


def test_load_train_reads_train_split_from_sample() -> None:
    with pytest.warns(UserWarning):
        out = list(load_train())
    assert len(out) == 1  # one train row in the committed sample
    assert out[0].gold == "math.PR"


def test_state_text_formats_title_and_abstract() -> None:
    with pytest.warns(UserWarning):
        ex = next(load(limit=1))
    assert ex.state_text.startswith("Title: ")
    assert "\nAbstract: " in ex.state_text


def test_meta_carries_example_id_and_publication_date() -> None:
    with pytest.warns(UserWarning):
        ex = next(load(limit=1))
    assert ex.meta["example_id"].startswith("arxiv-")
    assert date.fromisoformat(ex.meta["publication_date"]) > date.fromisoformat(BACKBONE_CUTOFF)


def test_limit_caps_output() -> None:
    with pytest.warns(UserWarning):
        assert len(list(load(limit=1))) == 1
