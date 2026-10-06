"""arXiv post-cutoff — anti-contamination benchmark. See README.md."""

from benchmarks.academic.arxiv_post_cutoff.loader import (
    BACKBONE_CUTOFF,
    CATEGORIES,
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)

__all__ = [
    "BACKBONE_CUTOFF",
    "CATEGORIES",
    "QUESTION",
    "REWORD_STEMS",
    "SOURCE",
    "load",
    "load_train",
]
