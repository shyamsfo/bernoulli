"""TweetEval-emotion — benchmark loader.

Dataset: `cardiffnlp/tweet_eval` config `emotion`. Tweets labeled with
one of 4 emotions: 0=anger, 1=joy, 2=optimism, 3=sadness. Framed as a
4-option `ChoiceQuestion`.

- `load(limit=None)` → `Iterator[BenchmarkExample]` over `test` (1,421 examples).
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train`
  (3,257 examples) for `BGEm3LR` and other trainable baselines.
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import ChoiceQuestion

SOURCE = "tweeteval_emotion"
_HF_PATH = "cardiffnlp/tweet_eval"
_HF_CONFIG = "emotion"
_LABEL_TO_OPTION = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness",
}

QUESTION = ChoiceQuestion(
    id="emotion",
    prompt="Which emotion best captures the feeling expressed in this tweet?",
    options=["anger", "joy", "optimism", "sadness"],
)

REWORD_STEMS: tuple[str, ...] = (
    "What emotion is the author of this tweet expressing?",
    "Classify the emotion in this tweet as anger, joy, optimism, or sadness.",
    "Which of these four emotions does the tweet convey most strongly?",
)


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, _HF_CONFIG, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield BenchmarkExample(
            state_text=row["text"].strip(),
            question=QUESTION,
            gold=_LABEL_TO_OPTION[row["label"]],
            source=SOURCE,
            meta={"example_id": f"tweeteval_emotion-{split}-{i}"},
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`test`, 1,421 examples)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (3,257 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)
