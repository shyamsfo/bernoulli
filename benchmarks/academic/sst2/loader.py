"""SST-2 (Stanford Sentiment Treebank, 2-class) — benchmark loader.

Validation split (872 examples) because GLUE's test labels are not public.
Framed as a 2-option ChoiceQuestion so Bernoulli answers with a label letter
rather than reading sentiment-token logits.

- Primary (eval): `load(limit=None)` → `Iterator[BenchmarkExample]` over `validation`.
- Training (for `BGEm3LR` and other trainable baselines):
  `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train` (67,349 examples).
- Reword stems: `REWORD_STEMS` — three paraphrased question stems used by the
  stability runner. Semantics preserved; wording varied.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion

SOURCE = "sst2"
_HF_PATH = "stanfordnlp/sst2"
_LABEL_TO_OPTION = {0: "negative", 1: "positive"}

QUESTION = ChoiceQuestion(
    id="sentiment",
    prompt="Is the sentiment of this excerpt positive or negative?",
    options=["negative", "positive"],
)

# Paraphrased stems for the stability test. Same intent, same answer space.
REWORD_STEMS: tuple[str, ...] = (
    "Does this excerpt convey a positive or a negative sentiment?",
    "Would you say the tone of this excerpt is positive or negative?",
    "Classify the sentiment of the excerpt below as either positive or negative.",
)


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield BenchmarkExample(
            state_text=row["sentence"].strip(),
            question=QUESTION,
            gold=_LABEL_TO_OPTION[row["label"]],
            source=SOURCE,
            meta={"example_id": f"sst2-{split}-{row['idx']}"},
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`validation`, 872 examples)."""
    return _yield_split("validation", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (67,349 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)
