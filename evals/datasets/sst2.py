"""SST-2 (Stanford Sentiment Treebank, 2-class) via HF datasets.

GLUE's SST-2 test labels are not public, so we use `validation` (872 examples).
Each example is a one-sentence excerpt labeled 0 (negative) or 1 (positive).
We frame as a 2-option ChoiceQuestion so the model answers with a label letter
rather than reading sentiment-token logits directly.
"""

from __future__ import annotations

from collections.abc import Iterator

from bernoulli.types import ChoiceQuestion
from evals.example import EvalExample

SOURCE = "sst2"
_LABEL_TO_OPTION = {0: "negative", 1: "positive"}
_QUESTION = ChoiceQuestion(
    id="sentiment",
    prompt="Is the sentiment of this excerpt positive or negative?",
    options=["negative", "positive"],
)


def load(limit: int | None = None) -> Iterator[EvalExample]:
    """Stream SST-2 validation examples as EvalExample.

    `limit` caps the number yielded (useful for smoke runs). None = full set.
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="validation")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield EvalExample(
            state_text=row["sentence"].strip(),
            question=_QUESTION,
            gold=_LABEL_TO_OPTION[row["label"]],
            source=SOURCE,
        )
