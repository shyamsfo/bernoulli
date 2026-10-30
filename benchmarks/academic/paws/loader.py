"""PAWS (Paraphrase Adversaries from Word Scrambling) — benchmark loader.

Dataset: `google-research-datasets/paws` config `labeled_final` (the
clean, hand-labeled split; `labeled_swap` is a related noisy variant).
Each example is a sentence pair labeled 1=paraphrase, 0=different.

First benchmark in the suite with a `BinaryQuestion`. The two sentences
are stitched into one `state_text` as:

    Sentence 1: ...
    Sentence 2: ...

so a single state carries both. Gold maps to `"Yes"` / `"No"` to match
the binary response-key convention in `bernoulli/labels.py`.

- `load(limit=None)` → `Iterator[BenchmarkExample]` over `test` (8,000 examples).
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train`
  (49,401 examples) for `BGEm3LR` and other trainable baselines.
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import BinaryQuestion

SOURCE = "paws"
_HF_PATH = "google-research-datasets/paws"
_HF_CONFIG = "labeled_final"
_LABEL_TO_OPTION = {0: "No", 1: "Yes"}

QUESTION = BinaryQuestion(
    id="paraphrase",
    prompt="Does Sentence 2 convey the same meaning as Sentence 1?",
)

REWORD_STEMS: tuple[str, ...] = (
    "Are these two sentences paraphrases of each other?",
    "Would you say Sentence 2 says the same thing as Sentence 1?",
    "Do the two sentences mean the same thing?",
)


def _format_pair(sentence1: str, sentence2: str) -> str:
    return f"Sentence 1: {sentence1.strip()}\nSentence 2: {sentence2.strip()}"


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, _HF_CONFIG, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield BenchmarkExample(
            state_text=_format_pair(row["sentence1"], row["sentence2"]),
            question=QUESTION,
            gold=_LABEL_TO_OPTION[row["label"]],
            source=SOURCE,
            meta={"example_id": f"paws-{split}-{row.get('id', i)}"},
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`test`, 8,000 examples)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (49,401 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)
