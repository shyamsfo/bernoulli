"""SST-2 — binary sentiment. See README.md in this folder."""

from benchmarks.academic.sst2.loader import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)

__all__ = ["QUESTION", "REWORD_STEMS", "SOURCE", "load", "load_train"]
