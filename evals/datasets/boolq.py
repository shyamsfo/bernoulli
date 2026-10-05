"""BoolQ (yes/no reading comprehension) via HF datasets.

Dataset: google/boolq. Split: validation (3270 examples). Each example has
a passage, a yes/no question about the passage, and a boolean answer.

Framed naturally as a BinaryQuestion with the passage as state and the
dataset's question as the question prompt. Gold is 'Yes' / 'No'.
"""

from __future__ import annotations

from collections.abc import Iterator

from bernoulli.types import BinaryQuestion
from evals.example import EvalExample

SOURCE = "boolq"


def load(limit: int | None = None) -> Iterator[EvalExample]:
    """Stream BoolQ validation examples as EvalExample."""
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield EvalExample(
            state_text=row["passage"].strip(),
            question=BinaryQuestion(
                id="answer",
                prompt=row["question"].strip() + "?",
            ),
            gold="Yes" if row["answer"] else "No",
            source=SOURCE,
        )
