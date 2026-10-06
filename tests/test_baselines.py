"""Generative baseline unit tests (parse logic + prompt shape, no GPU)."""

from __future__ import annotations

import pytest

from benchmarks.common.baselines import (
    _build_generative_prompt,
    _decide_one_generative,
    generative_decide,
    parse_response,
)
from bernoulli.types import BinaryQuestion, ChoiceQuestion, DecideRequest, RatingQuestion, State


class TestParseResponse:
    def test_exact_match_first_option(self) -> None:
        assert parse_response("refund", ["refund", "exchange", "tracking"]) == 0

    def test_match_within_sentence(self) -> None:
        assert parse_response("The answer is tracking.", ["refund", "exchange", "tracking"]) == 2

    def test_case_insensitive(self) -> None:
        assert parse_response("REFUND", ["refund", "exchange"]) == 0

    def test_first_by_position_wins(self) -> None:
        """If multiple options appear in the text, the one at the earliest position wins."""
        assert parse_response("other, probably refund", ["refund", "other"]) == 1

    def test_no_match_returns_none(self) -> None:
        assert parse_response("I don't know", ["refund", "exchange"]) is None

    def test_empty_text(self) -> None:
        assert parse_response("", ["a", "b"]) is None

    def test_binary_yes_no(self) -> None:
        assert parse_response("Yes", ["Yes", "No"]) == 0
        assert parse_response("No", ["Yes", "No"]) == 1

    def test_rating_integer_strings(self) -> None:
        assert parse_response("4", ["1", "2", "3", "4", "5"]) == 3


class TestBuildPrompt:
    def test_includes_options_in_system(self, hf_tokenizer: object) -> None:
        q = ChoiceQuestion(id="q", prompt="which?", options=["apple", "banana"])
        prompt = _build_generative_prompt(hf_tokenizer, "state", q)
        assert "apple" in prompt
        assert "banana" in prompt
        assert "classifier" in prompt.lower()

    def test_binary_options_rendered_as_yes_no(self, hf_tokenizer: object) -> None:
        q = BinaryQuestion(id="q", prompt="urgent?")
        prompt = _build_generative_prompt(hf_tokenizer, "state", q)
        assert "Yes" in prompt
        assert "No" in prompt

    def test_rating_options_rendered_as_integer_strings(self, hf_tokenizer: object) -> None:
        q = RatingQuestion(id="q", prompt="?", scale=(1, 5))
        prompt = _build_generative_prompt(hf_tokenizer, "state", q)
        for v in ["1", "2", "3", "4", "5"]:
            assert v in prompt


class _ScriptedGenerator:
    """Mock HFScorer-shaped object that returns a fixed generated response."""

    model_id = "test"
    revision = None

    def __init__(self, tokenizer: object, response: str) -> None:
        self.tokenizer = tokenizer
        self._response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int = 10) -> str:
        self.calls.append(prompt)
        return self._response


class TestDecideOneGenerative:
    def test_choice_parses_from_text(self, hf_tokenizer: object) -> None:
        q = ChoiceQuestion(id="intent", prompt="what?", options=["refund", "exchange"])
        scorer = _ScriptedGenerator(hf_tokenizer, "The answer is refund.")
        decision = _decide_one_generative(scorer, "state", q)
        assert decision.type == "choice"
        assert decision.answer == "refund"
        assert decision.confidence == 1.0
        assert decision.distribution == {"refund": 1.0, "exchange": 0.0}

    def test_choice_uniform_fallback_on_unparseable(self, hf_tokenizer: object) -> None:
        q = ChoiceQuestion(
            id="intent", prompt="?", options=["refund", "exchange", "tracking", "complaint"]
        )
        # Response that mentions none of the options → parser returns None → uniform.
        scorer = _ScriptedGenerator(hf_tokenizer, "I'm not sure.")
        decision = _decide_one_generative(scorer, "state", q)
        assert all(abs(p - 0.25) < 1e-6 for p in decision.distribution.values())

    def test_binary_yes(self, hf_tokenizer: object) -> None:
        q = BinaryQuestion(id="urgent", prompt="urgent?")
        scorer = _ScriptedGenerator(hf_tokenizer, "Yes, definitely.")
        decision = _decide_one_generative(scorer, "state", q)
        assert decision.type == "binary"
        assert decision.answer is True
        assert decision.probability == 1.0

    def test_rating_four(self, hf_tokenizer: object) -> None:
        q = RatingQuestion(id="anger", prompt="?", scale=(1, 5))
        scorer = _ScriptedGenerator(hf_tokenizer, "4")
        decision = _decide_one_generative(scorer, "state", q)
        assert decision.type == "rating"
        assert decision.expected == pytest.approx(4.0)


class TestGenerativeDecideEnd2End:
    def test_full_request_response(self, hf_tokenizer: object) -> None:
        q = ChoiceQuestion(id="intent", prompt="?", options=["refund", "other"])
        req = DecideRequest(state=State(text="give me my money back"), questions=[q])
        scorer = _ScriptedGenerator(hf_tokenizer, "refund")
        resp = generative_decide(req, scorer)
        assert resp.decisions["intent"].type == "choice"
        assert resp.decisions["intent"].answer == "refund"
        assert resp.model == "test"
        assert resp.latency_ms >= 0
