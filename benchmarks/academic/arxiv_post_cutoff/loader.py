"""arXiv post-cutoff — benchmark loader.

4-way primary-category classification over arXiv papers published
**strictly after** the backbone model's training cutoff. The point is
anti-contamination: the model cannot have seen these papers at pretrain
time, so classification performance reflects generalisation rather than
memorisation.

- Categories: `cs.CL`, `cs.CV`, `math.PR`, `econ.EM` — chosen to be
  semantically distinct (NLP vs vision vs probability vs econometrics)
  so a competent classifier is distinguishable from a bag-of-words
  shortcut.
- Cutoff: see `BACKBONE_CUTOFF` constant. Default is `2025-01-01`,
  safely after Qwen2.5-VL's pretraining cutoff (late 2023). Any paper
  with `publication_date >= BACKBONE_CUTOFF` is eligible.
- Dataset lives at `data/dataset.jsonl` (one example per line) once
  `make_dataset.py` has been run. A 3-row `data/sample.jsonl` is
  committed for tests + leaderboard-structure validation; if
  `data/dataset.jsonl` is missing the loader falls back to the sample
  and emits a warning.

- `load(limit=None)` → `Iterator[BenchmarkExample]` over the eval set.
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over the
  train set. May be empty if the full dataset hasn't been generated
  yet; `BGEm3LR.fit` will raise `ValueError` in that case, which the
  runner handles by skipping the baseline column.
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion

SOURCE = "arxiv_post_cutoff"
BACKBONE_CUTOFF = "2025-01-01"
CATEGORIES: tuple[str, ...] = ("cs.CL", "cs.CV", "math.PR", "econ.EM")

QUESTION = ChoiceQuestion(
    id="arxiv_category",
    prompt=(
        "What is the primary arXiv subject category for this paper? "
        "Choose cs.CL (NLP), cs.CV (computer vision), math.PR (probability), "
        "or econ.EM (econometrics)."
    ),
    options=list(CATEGORIES),
)

REWORD_STEMS: tuple[str, ...] = (
    "Which arXiv primary category does this paper belong to: cs.CL, cs.CV, math.PR, or econ.EM?",
    "Classify this arXiv paper by primary subject category.",
    "Based on title and abstract, pick the best-fitting arXiv primary category.",
)

_DATA_DIR = Path(__file__).parent / "data"
_FULL_PATH = _DATA_DIR / "dataset.jsonl"
_SAMPLE_PATH = _DATA_DIR / "sample.jsonl"


def _resolve_source_file(split: str) -> Path:
    """Pick the file to read: full dataset if present, otherwise warn + sample."""
    if _FULL_PATH.exists():
        return _FULL_PATH
    warnings.warn(
        f"arxiv_post_cutoff: {_FULL_PATH.name} not found — falling back to "
        f"{_SAMPLE_PATH.name} ({split=}). Run `python -m "
        f"benchmarks.academic.arxiv_post_cutoff.make_dataset` on the dev box "
        f"to generate the full post-cutoff snapshot.",
        stacklevel=3,
    )
    return _SAMPLE_PATH


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    """Yield examples whose `split` field matches. Rows without a `split` field default to `test`."""
    source_file = _resolve_source_file(split)
    with source_file.open() as fp:
        yielded = 0
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "test") != split:
                continue
            if limit is not None and yielded >= limit:
                break
            if row["primary_category"] not in CATEGORIES:
                # Defensive: skip any row that slipped in with an out-of-vocab category.
                continue
            state_text = f"Title: {row['title'].strip()}\nAbstract: {row['abstract'].strip()}"
            yield BenchmarkExample(
                state_text=state_text,
                question=QUESTION,
                gold=row["primary_category"],
                source=SOURCE,
                meta={
                    "example_id": f"arxiv-{row['id']}",
                    "publication_date": row["publication_date"],
                },
            )
            yielded += 1


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (rows with `split == "test"`)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (rows with `split == "train"`) — may be empty on the sample file."""
    return _yield_split("train", limit)
