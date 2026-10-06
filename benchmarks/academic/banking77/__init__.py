"""Banking77 — 77-way intent classification. See README.md in this folder.

Deviation from the sst2/ag_news shape: Banking77's option list has 77
entries that are not stable across dataset revisions worth typing by
hand, so `QUESTION` is a function (`question()`) that builds the
`ChoiceQuestion` lazily by walking the dataset. `load()` and `load_train()`
reuse the same construction internally; external callers (stability
runner, DeBERTa adapter) that need the question can call `question()`.
"""

from benchmarks.academic.banking77.loader import (
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
    question,
)

__all__ = ["REWORD_STEMS", "SOURCE", "load", "load_train", "question"]
