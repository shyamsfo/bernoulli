"""Prompt builder — chat template + state-first/question-last layout.

Vision doc §4 layout:
    system: "You are a decision function. Answer with a single letter."
    user:   <state text> <question> <labeled options>
    assistant prefill: "Answer:" so the next token is the label.

State-first/question-last ordering is deliberate: when multiple questions ask
about the same state, the state tokens form a shared prefix that engine-level
prefix caching can reuse. Image handling lands in M4 (text-only here).
"""

from __future__ import annotations

from typing import Any, Protocol

from bernoulli.labels import BINARY_NO, BINARY_YES, LETTERS
from bernoulli.types import BinaryQuestion, ChoiceQuestion, Question, RatingQuestion

SYSTEM_PROMPT = "You are a decision function. Answer with a single letter."
ASSISTANT_PREFILL = "Answer:"


class _ChatTemplater(Protocol):
    """Minimal tokenizer surface for the chat template."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> str: ...


def labels_for(question: Question) -> list[str]:
    """Return the letter labels this question maps to, in canonical order.

    - choice: A..{len(options)}
    - binary: ['Yes', 'No'] (not letters — Yes/No are single-token on target
      tokenizers and more interpretable than 'A/B')
    - rating: A..{scale_width} (A = scale low, B = low+1, ...)
    """
    if isinstance(question, ChoiceQuestion):
        return list(LETTERS[: len(question.options)])
    if isinstance(question, BinaryQuestion):
        return [BINARY_YES, BINARY_NO]
    if isinstance(question, RatingQuestion):
        low, high = question.scale
        return list(LETTERS[: high - low + 1])
    raise TypeError(f"unknown question kind: {type(question).__name__}")


def option_strings_for(question: Question) -> list[str]:
    """Human-readable option text matched 1:1 with labels_for(question).

    - choice: the caller-supplied option strings
    - binary: ['Yes', 'No']
    - rating: the integer labels as strings, e.g. ['1', '2', '3', '4', '5']
    """
    if isinstance(question, ChoiceQuestion):
        return list(question.options)
    if isinstance(question, BinaryQuestion):
        return [BINARY_YES, BINARY_NO]
    if isinstance(question, RatingQuestion):
        low, high = question.scale
        return [str(i) for i in range(low, high + 1)]
    raise TypeError(f"unknown question kind: {type(question).__name__}")


def _render_options(labels: list[str], options: list[str]) -> str:
    return "\n".join(f"{label}) {opt}" for label, opt in zip(labels, options, strict=True))


def build_prompt(
    tok: _ChatTemplater,
    *,
    state_text: str,
    question: Question,
    option_order: list[int] | None = None,
) -> str:
    """Return the full prompt string, ready to tokenize for one forward pass.

    Labels (A, B, C, ... or Yes/No) are always rendered in canonical order at
    label positions 0, 1, 2, …. `option_order` is a permutation of
    range(num_options) describing which canonical option appears at each
    label position. None = identity (canonical rendering).

    The debiaser rerun the same request with different option_orders and
    averages the probabilities after mapping each run back to the canonical
    option index.
    """
    labels = labels_for(question)
    canonical_options = option_strings_for(question)
    if option_order is None:
        options = canonical_options
    else:
        if sorted(option_order) != list(range(len(canonical_options))):
            raise ValueError(
                f"option_order {option_order} is not a permutation of "
                f"range({len(canonical_options)})"
            )
        options = [canonical_options[i] for i in option_order]
    user_turn = f"{state_text}\n\n{question.prompt}\n{_render_options(labels, options)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_turn},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + ASSISTANT_PREFILL
