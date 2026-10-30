"""Smoke tests for benchmarks/academic/arxiv_post_cutoff/.

Reads from the committed real `data/dataset.jsonl` (generated 2026-10-07 by
`make_dataset.py`). Covers schema, shape of the yielded examples, and the
anti-contamination invariant (every row's publication_date > BACKBONE_CUTOFF).

The sample-fallback warning path (dataset.jsonl absent → sample.jsonl +
UserWarning) is covered by monkey-patching `_FULL_PATH` so we don't have to
move the committed dataset aside.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import benchmarks.academic.arxiv_post_cutoff.loader as arxiv_loader
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
_DATASET_FILE = _SAMPLE_FILE.with_name("dataset.jsonl")


def test_sample_file_exists() -> None:
    assert _SAMPLE_FILE.exists(), f"sample dataset must be committed at {_SAMPLE_FILE}"


def test_dataset_file_exists() -> None:
    assert _DATASET_FILE.exists(), f"real dataset must be committed at {_DATASET_FILE}"


def test_load_reads_test_split() -> None:
    out = list(load())
    assert len(out) >= 50
    assert all(isinstance(ex, BenchmarkExample) for ex in out)
    assert all(ex.source == SOURCE for ex in out)
    # Every row's gold must be one of our four canonical categories.
    assert {ex.gold for ex in out}.issubset(set(CATEGORIES))


def test_load_train_reads_train_split() -> None:
    out = list(load_train())
    assert len(out) >= 50
    # Train split should include every category.
    assert {ex.gold for ex in out}.issubset(set(CATEGORIES))


def test_state_text_formats_title_and_abstract() -> None:
    ex = next(load(limit=1))
    assert ex.state_text.startswith("Title: ")
    assert "\nAbstract: " in ex.state_text


def test_meta_carries_example_id_and_publication_date() -> None:
    ex = next(load(limit=1))
    assert ex.meta["example_id"].startswith("arxiv-")
    assert date.fromisoformat(ex.meta["publication_date"]) > date.fromisoformat(BACKBONE_CUTOFF)


def test_every_test_row_is_post_cutoff() -> None:
    """Anti-contamination invariant: no row's publication_date <= BACKBONE_CUTOFF."""
    cutoff = date.fromisoformat(BACKBONE_CUTOFF)
    for ex in load():
        pub = date.fromisoformat(ex.meta["publication_date"])
        assert pub > cutoff, f"{ex.meta['example_id']} violates cutoff: {pub} <= {cutoff}"


def test_limit_caps_output() -> None:
    assert len(list(load(limit=5))) == 5


def test_fallback_warns_when_dataset_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fallback path: pretend dataset.jsonl is missing → warn + use sample."""
    monkeypatch.setattr(arxiv_loader, "_FULL_PATH", tmp_path / "nonexistent.jsonl")
    with pytest.warns(UserWarning, match="not found .* falling back"):
        out = list(load())
    # Sample has 2 test rows (cs.CL + cs.CV).
    assert len(out) == 2
    assert {ex.gold for ex in out} == {"cs.CL", "cs.CV"}
