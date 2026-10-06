"""PAWS — paraphrase detection (binary). See README.md in this folder."""

from benchmarks.academic.paws.loader import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)

__all__ = ["QUESTION", "REWORD_STEMS", "SOURCE", "load", "load_train"]
