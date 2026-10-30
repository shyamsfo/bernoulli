"""Banking77 (77-way intent classification) via HF datasets.

Dataset: mteb/banking77. Split: test (3076 examples). Each example is a
customer utterance labeled with one of 77 banking intents (card_arrival,
refund_not_showing_up, visa_or_mastercard, ...).

Framed as a 77-option ChoiceQuestion. Exceeds the 26-letter alphabet, so
bernoulli/chunked.py handles the aggregation (3 forward passes per decision).
The letter alphabet constraint is called out in vision_and_roadmap.md §9.

Note on dataset choice: the original PolyAI/banking77 upload uses the
deprecated HF dataset-script format and no longer loads under
`datasets >= 3`. mteb/banking77 is the MTEB reupload in parquet with
identical semantics plus a convenient per-example `label_text`.
"""

from __future__ import annotations

from collections.abc import Iterator

from bernoulli.types import ChoiceQuestion
from evals.example import EvalExample

SOURCE = "banking77"


def _ordered_label_names(ds: object) -> list[str]:
    """Build the canonical option order by walking the dataset once.

    mteb/banking77 ships int `label` + string `label_text`. We collect the
    int->text mapping and sort by int so the option order is reproducible
    across environments (not dependent on dataset iteration order).
    """
    label_to_text: dict[int, str] = {}
    for row in ds:  # type: ignore[attr-defined]
        label_to_text[row["label"]] = row["label_text"]
    return [label_to_text[i] for i in sorted(label_to_text)]


def load(limit: int | None = None) -> Iterator[EvalExample]:
    """Stream Banking77 test examples as EvalExample."""
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77", split="test")
    options = _ordered_label_names(ds)
    question = ChoiceQuestion(
        id="intent",
        prompt="Which banking intent best matches the customer's message?",
        options=options,
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield EvalExample(
            state_text=row["text"].strip(),
            question=question,
            gold=row["label_text"],
            source=SOURCE,
        )
