"""AG News — 4-way topic classification. See README.md in this folder."""

from benchmarks.academic.ag_news.loader import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)

__all__ = ["QUESTION", "REWORD_STEMS", "SOURCE", "load", "load_train"]
