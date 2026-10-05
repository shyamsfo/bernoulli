"""The common (state, question, gold) format every dataset loader produces.

A benchmark dataset gets converted into a stream of EvalExample. The eval
harness then calls decide() against each example's question and compares
the predicted distribution to the gold label.
"""

from __future__ import annotations

from dataclasses import dataclass

from bernoulli.types import Question


@dataclass(frozen=True)
class EvalExample:
    """One benchmark example in the common format.

    The gold field is the option string the model should pick, matched to
    the response distribution keys:
    - choice: one of question.options
    - binary: 'Yes' or 'No'
    - rating: integer label as string, e.g. '3' for scale (1, 5)
    """

    state_text: str
    question: Question
    gold: str
    source: str = ""  # dataset tag, e.g. 'sst2', for reporting
