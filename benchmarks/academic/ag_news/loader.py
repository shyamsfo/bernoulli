"""AG News (4-way topic classification) — benchmark loader.

Dataset: `fancyzhx/ag_news`. Short news snippets labeled 0=World,
1=Sports, 2=Business, 3=Sci/Tech. Framed as a 4-option ChoiceQuestion;
the letter alphabet (A-D) handles this comfortably.

- `load(limit=None)` → `Iterator[BenchmarkExample]` over `test` (7,600 examples).
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train`
  (120,000 examples) for `BGEm3LR` and other trainable baselines.
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion

SOURCE = "ag_news"
_HF_PATH = "fancyzhx/ag_news"
_LABEL_TO_OPTION = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

QUESTION = ChoiceQuestion(
    id="topic",
    prompt="Which topic does this news article belong to?",
    options=["World", "Sports", "Business", "Sci/Tech"],
)

REWORD_STEMS: tuple[str, ...] = (
    "What is the primary topic of this news article?",
    "Classify this news article by topic.",
    "Which of these categories best fits the article: World, Sports, Business, or Sci/Tech?",
)


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield BenchmarkExample(
            state_text=row["text"].strip(),
            question=QUESTION,
            gold=_LABEL_TO_OPTION[row["label"]],
            source=SOURCE,
            meta={"example_id": f"ag_news-{split}-{i}"},
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`test`, 7,600 examples)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (120,000 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)
