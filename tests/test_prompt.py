"""Prompt-builder shape tests. Uses the real Qwen tokenizer (chat template)
but no GPU."""

from __future__ import annotations

from bernoulli.prompt import (
    ASSISTANT_PREFILL,
    build_prompt,
    labels_for,
    option_strings_for,
)
from bernoulli.types import BinaryQuestion, ChoiceQuestion, RatingQuestion


class TestLabelsFor:
    def test_choice_uses_letters_matching_option_count(self) -> None:
        q = ChoiceQuestion(id="q", prompt="?", options=["a", "b", "c"])
        assert labels_for(q) == ["A", "B", "C"]

    def test_binary_uses_yes_no(self) -> None:
        q = BinaryQuestion(id="q", prompt="?")
        assert labels_for(q) == ["Yes", "No"]

    def test_rating_uses_letters_matching_scale_width(self) -> None:
        q = RatingQuestion(id="q", prompt="?", scale=(1, 5))
        assert labels_for(q) == ["A", "B", "C", "D", "E"]

    def test_rating_option_strings_are_integer_strings(self) -> None:
        q = RatingQuestion(id="q", prompt="?", scale=(1, 5))
        assert option_strings_for(q) == ["1", "2", "3", "4", "5"]


class TestBuildPrompt:
    def test_choice_prompt_ends_with_prefill(self, hf_tokenizer: object) -> None:
        q = ChoiceQuestion(id="q", prompt="which?", options=["apple", "banana"])
        prompt = build_prompt(hf_tokenizer, state_text="customer wrote: hello", question=q)
        assert prompt.endswith(ASSISTANT_PREFILL)
        assert "A) apple" in prompt
        assert "B) banana" in prompt
        assert "customer wrote: hello" in prompt

    def test_binary_renders_yes_no(self, hf_tokenizer: object) -> None:
        q = BinaryQuestion(id="q", prompt="urgent?")
        prompt = build_prompt(hf_tokenizer, state_text="x", question=q)
        assert "Yes) Yes" in prompt
        assert "No) No" in prompt

    def test_rating_renders_integer_options(self, hf_tokenizer: object) -> None:
        q = RatingQuestion(id="q", prompt="how angry?", scale=(1, 5))
        prompt = build_prompt(hf_tokenizer, state_text="x", question=q)
        for label, value in zip("ABCDE", "12345", strict=True):
            assert f"{label}) {value}" in prompt

    def test_canonical_ordering_is_stable(self, hf_tokenizer: object) -> None:
        q = ChoiceQuestion(id="q", prompt="?", options=["apple", "banana", "cherry"])
        prompt = build_prompt(hf_tokenizer, state_text="x", question=q)
        # Canonical rendering is always A→first option, B→second, etc.
        a_idx = prompt.index("A) apple")
        b_idx = prompt.index("B) banana")
        c_idx = prompt.index("C) cherry")
        assert a_idx < b_idx < c_idx
