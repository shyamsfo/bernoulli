"""Banking77 (77-way intent classification) — benchmark loader.

Dataset: `mteb/banking77` (parquet reupload of the original PolyAI set;
the PolyAI dataset-script format no longer loads under `datasets>=3`).

Customer utterances labeled with one of 77 banking intents
(`card_arrival`, `refund_not_showing_up`, `visa_or_mastercard`, ...).
Framed as a 77-option `ChoiceQuestion` — exceeds the 26-letter alphabet
so `bernoulli/chunked.py` kicks in for scoring (3 forward passes per
decision).

- `load(limit=None)` → `Iterator[BenchmarkExample]` over `test` (3,076 examples).
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train`
  (10,003 examples) for `BGEm3LR` and other trainable baselines.
- `question()` builds the `ChoiceQuestion` by walking the eval split
  once to collect the canonical int→text label mapping (sorted by int
  for reproducibility).
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion

SOURCE = "banking77"
_HF_PATH = "mteb/banking77"
_PROMPT = "Which banking intent best matches the customer's message?"

REWORD_STEMS: tuple[str, ...] = (
    "Classify the customer's message by banking intent.",
    "What is the customer asking about?",
    "Pick the banking intent that best describes the customer's message.",
)


def _options_from_dataset(ds: object) -> list[str]:
    """Collect the int->text label mapping and return a list sorted by int.

    Sorting by the integer label makes the option order reproducible
    across environments regardless of row iteration order.
    """
    label_to_text: dict[int, str] = {}
    for row in ds:  # type: ignore[attr-defined]
        label_to_text[row["label"]] = row["label_text"]
    return [label_to_text[i] for i in sorted(label_to_text)]


def question() -> ChoiceQuestion:
    """Build the Banking77 `ChoiceQuestion` by walking the test split for labels."""
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, split="test")
    return ChoiceQuestion(id="intent", prompt=_PROMPT, options=_options_from_dataset(ds))


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, split=split)
    options = _options_from_dataset(ds)
    q = ChoiceQuestion(id="intent", prompt=_PROMPT, options=options)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield BenchmarkExample(
            state_text=row["text"].strip(),
            question=q,
            gold=row["label_text"],
            source=SOURCE,
            meta={"example_id": f"banking77-{split}-{i}"},
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`test`, 3,076 examples)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (10,003 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)
