"""Yelp 1-5 star reviews — benchmark loader.

Dataset: `yelp_review_full`. Half a million Yelp review paragraphs
labeled 0..4 (= 1..5 stars). This is the first benchmark in the suite
that uses `RatingQuestion` — exercises the integer-string response-key
convention (`"1"` .. `"5"`) and the expected-value aggregation
machinery.

- `load(limit=None)` → `Iterator[BenchmarkExample]` over `test` (50,000 examples).
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train`
  (650,000 examples) for `BGEm3LR` and other trainable baselines.
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import RatingQuestion

SOURCE = "yelp_stars"
_HF_PATH = "yelp_review_full"
_SCALE = (1, 5)

QUESTION = RatingQuestion(
    id="stars",
    prompt="On a 1-5 scale, how many stars best represent this review's sentiment?",
    scale=_SCALE,
)

REWORD_STEMS: tuple[str, ...] = (
    "Rate the sentiment of this review on a 1-5 scale.",
    "How many stars (1 to 5) would best match this review?",
    "Pick the number of stars from 1 to 5 that fits this review.",
)


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        # label is 0..4; the response-key convention is "1".."5".
        gold = str(int(row["label"]) + _SCALE[0])
        yield BenchmarkExample(
            state_text=row["text"].strip(),
            question=QUESTION,
            gold=gold,
            source=SOURCE,
            meta={"example_id": f"yelp-{split}-{i}"},
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`test`, 50,000 examples)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (650,000 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)
