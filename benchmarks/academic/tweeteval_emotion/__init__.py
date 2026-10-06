"""TweetEval-emotion — 4-way emotion classification over tweets. See README.md."""

from benchmarks.academic.tweeteval_emotion.loader import (
    QUESTION,
    REWORD_STEMS,
    SOURCE,
    load,
    load_train,
)

__all__ = ["QUESTION", "REWORD_STEMS", "SOURCE", "load", "load_train"]
