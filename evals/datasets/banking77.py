"""Banking77 (77-way intent classification) via HF datasets.

Dataset: PolyAI/banking77. Split: test (3080 examples). Each example is a
customer utterance labeled with one of 77 banking intents (card_arrival,
refund_not_showing_up, visa_or_mastercard, ...).

Framed as a 77-option ChoiceQuestion. Exceeds the 26-letter alphabet, so
bernoulli/chunked.py handles the aggregation (3 forward passes per decision).
The letter alphabet constraint is called out in vision_and_roadmap.md §9.
"""

from __future__ import annotations

from collections.abc import Iterator

from bernoulli.types import ChoiceQuestion
from evals.example import EvalExample

SOURCE = "banking77"


def _label_names() -> list[str]:
    """Load the canonical 77-label list from the dataset metadata.

    Lazy so this module doesn't require `datasets` at import time.
    """
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77", split="test")
    return list(ds.features["label"].names)


def load(limit: int | None = None) -> Iterator[EvalExample]:
    """Stream Banking77 test examples as EvalExample."""
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77", split="test")
    label_names: list[str] = list(ds.features["label"].names)
    question = ChoiceQuestion(
        id="intent",
        prompt="Which banking intent best matches the customer's message?",
        options=label_names,
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield EvalExample(
            state_text=row["text"].strip(),
            question=question,
            gold=label_names[row["label"]],
            source=SOURCE,
        )
