"""Label-token discovery and single-token assertion.

Every label used in the prompt must encode to exactly one token for the chosen
backbone's tokenizer. If it doesn't, the "read the next-token logit at the
answer position" scheme breaks — we'd be looking at a prefix token, not the
label itself.

Dev-backbone finding (Qwen2.5-VL tokenizer, Qwen2Tokenizer):
- Letters ' A'..' Z' with leading space: single token each.
- Digits ' 1'..' 9' with leading space: TWO tokens each (space + digit).
- 'Yes' / 'No' with leading space: single token each.

So we use letters uniformly across all question types. Ratings map
A -> low, B -> low+1, ..., up to scale width <= 9 (A..I). This is a deviation
from vision_and_roadmap.md §4 which assumed digits; it's recorded in CLAUDE.md.
"""

from __future__ import annotations

from typing import Protocol

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BINARY_YES = "Yes"
BINARY_NO = "No"


class _HasEncode(Protocol):
    """Minimal tokenizer surface. Any HF tokenizer satisfies this."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def _single_token(tok: _HasEncode, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"label {text!r} encodes to {len(ids)} tokens on this tokenizer; "
            f"the scorer requires single-token labels. ids={ids}"
        )
    return ids[0]


def letter_token_ids(tok: _HasEncode) -> dict[str, int]:
    """{'A': id, 'B': id, ...} for all 26 letters, space-prefixed under the hood.

    Fails loudly if any letter doesn't encode to exactly one token.
    """
    return {letter: _single_token(tok, " " + letter) for letter in LETTERS}


def binary_token_ids(tok: _HasEncode) -> dict[str, int]:
    """{'Yes': id, 'No': id}, space-prefixed under the hood."""
    return {
        BINARY_YES: _single_token(tok, " " + BINARY_YES),
        BINARY_NO: _single_token(tok, " " + BINARY_NO),
    }


def assert_single_token_labels(tok: _HasEncode) -> None:
    """Verify every required label is single-token on this tokenizer.

    Call once at scorer construction. Raises ValueError on the first failure
    with a message naming the offending label and its token count.
    """
    letter_token_ids(tok)
    binary_token_ids(tok)
