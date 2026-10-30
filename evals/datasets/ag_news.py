"""AG News (4-way topic classification) via HF datasets.

Dataset: fancyzhx/ag_news. Split: test (7600 examples). Each example is a
short news snippet labeled 0=World, 1=Sports, 2=Business, 3=Sci/Tech.

Framed as a 4-option ChoiceQuestion. The letter alphabet handles this
comfortably (A-D).
"""

from __future__ import annotations

from collections.abc import Iterator

from bernoulli.types import ChoiceQuestion
from evals.example import EvalExample

SOURCE = "ag_news"
_LABEL_TO_OPTION = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}
_QUESTION = ChoiceQuestion(
    id="topic",
    prompt="Which topic does this news article belong to?",
    options=["World", "Sports", "Business", "Sci/Tech"],
)


def load(limit: int | None = None) -> Iterator[EvalExample]:
    """Stream AG News test examples as EvalExample."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="test")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield EvalExample(
            state_text=row["text"].strip(),
            question=_QUESTION,
            gold=_LABEL_TO_OPTION[row["label"]],
            source=SOURCE,
        )
