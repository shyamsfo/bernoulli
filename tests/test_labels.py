"""Label-token encoding checks on the dev-tier backbone's tokenizer.

Needs the real tokenizer but no GPU — fast enough to run in the default CI.
"""

from __future__ import annotations

import pytest

from bernoulli.labels import (
    LETTERS,
    assert_single_token_labels,
    binary_token_ids,
    letter_token_ids,
)


def test_all_letters_single_token(hf_tokenizer: object) -> None:
    mapping = letter_token_ids(hf_tokenizer)  # type: ignore[arg-type]
    assert set(mapping.keys()) == set(LETTERS)
    assert len(set(mapping.values())) == 26  # all distinct


def test_binary_single_token(hf_tokenizer: object) -> None:
    mapping = binary_token_ids(hf_tokenizer)  # type: ignore[arg-type]
    assert set(mapping.keys()) == {"Yes", "No"}


def test_assert_single_token_labels_passes(hf_tokenizer: object) -> None:
    assert_single_token_labels(hf_tokenizer)  # type: ignore[arg-type]


def test_assert_single_token_labels_fails_on_multi_token() -> None:
    """A tokenizer that returns 2 tokens for ' A' should trip the assertion."""

    class BadTok:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [1, 2] if text == " A" else [1]

    with pytest.raises(ValueError, match="encodes to 2 tokens"):
        letter_token_ids(BadTok())
