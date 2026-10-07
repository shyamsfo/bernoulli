"""Debias permutation + averaging tests (no GPU needed — scorer is mocked)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from bernoulli.debias import _permutations_for, debias
from bernoulli.labels import letter_token_ids
from bernoulli.types import BinaryQuestion, ChoiceQuestion, RatingQuestion


class TestPermutations:
    def test_none_is_identity(self) -> None:
        assert _permutations_for("none", 3) == [[0, 1, 2]]

    def test_reverse(self) -> None:
        assert _permutations_for("reverse", 3) == [[0, 1, 2], [2, 1, 0]]

    def test_reverse_handles_singleton(self) -> None:
        assert _permutations_for("reverse", 1) == [[0]]

    def test_cyclic(self) -> None:
        perms = _permutations_for("cyclic", 3)
        assert perms == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    def test_cyclic_rejects_too_many_options(self) -> None:
        with pytest.raises(ValueError, match="capped at k <= 6"):
            _permutations_for("cyclic", 7)


class _FakeTokenizer:
    """Minimum tokenizer surface for debias()."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        # Deterministic single-token encoding for letters, Yes/No; raise otherwise
        if text.startswith(" ") and len(text) == 2 and text[1].isalpha():
            return [1000 + ord(text[1])]
        if text == " Yes":
            return [1200]
        if text == " No":
            return [1201]
        return [0, 0]  # multi-token, would fail labels assertion

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> str:
        # Return the user content verbatim so option order is observable in the prompt
        user_turn = next(m["content"] for m in conversation if m["role"] == "user")
        return str(user_turn)


class _ScriptedScorer:
    """Returns logits scripted by prompt content so each permutation gives
    a known distribution we can verify after debias averages them."""

    model_id = "test"
    revision = None

    def __init__(self, tokenizer: object, logits_by_first_option: dict[str, list[float]]) -> None:
        self.tokenizer = tokenizer
        self._logits_by_first_option = logits_by_first_option
        self.calls: list[str] = []

    def score(
        self,
        prompt: str,
        allowed_token_ids: list[int],
        *,
        images: list[str] | None = None,
    ) -> np.ndarray:
        self.calls.append(prompt)
        # Parse the first rendered option (whatever label precedes it) — that's
        # the permutation key.
        match = re.search(r"\n([A-Za-z]+)\) ([^\n]+)", prompt)
        if match is None:
            raise AssertionError(f"could not find first option in prompt:\n{prompt}")
        first_opt = match.group(2).strip()
        if first_opt not in self._logits_by_first_option:
            raise AssertionError(f"no scripted logits for first option {first_opt!r}")
        return np.asarray(self._logits_by_first_option[first_opt], dtype=np.float32)

    def score_batch(
        self,
        prompts: list[str],
        allowed_token_ids_list: list[list[int]],
        *,
        images_list: list[list[str] | None] | None = None,
    ) -> list[np.ndarray]:
        return [self.score(p, a) for p, a in zip(prompts, allowed_token_ids_list, strict=True)]


class TestDebiasReverse:
    def test_reverse_averages_canonical_and_reversed(self) -> None:
        """Scripted model gives P(label A)=0.9 in both runs. After debias,
        the canonical option at index 0 should get P ≈ (0.9 + 0.1) / 2 = 0.5
        because the second run has 'option 0' rendered at label B."""
        q = ChoiceQuestion(id="q", prompt="?", options=["apple", "banana"])
        tok = _FakeTokenizer()
        scorer = _ScriptedScorer(
            tok,
            {
                # Run 1: options rendered [apple, banana] at [A, B]. P(A)=0.9.
                "apple": [10.0, 7.8],  # softmax ≈ [0.9, 0.1]
                # Run 2: options rendered [banana, apple] at [A, B]. P(A)=0.9 again.
                "banana": [10.0, 7.8],
            },
        )
        probs = debias(scorer, state_text="x", question=q, mode="reverse")
        # Canonical option 0 (apple) got P=0.9 in run 1 (position A) and P=0.1 in run 2 (position B)
        # Averaged → 0.5
        assert np.isclose(probs[0], 0.5, atol=0.01)
        assert np.isclose(probs[1], 0.5, atol=0.01)
        assert len(scorer.calls) == 2  # one call per permutation

    def test_reverse_amplifies_true_signal(self) -> None:
        """When the model answers correctly regardless of position, the debiased
        probability for the correct option is close to the per-run probability."""
        q = ChoiceQuestion(id="q", prompt="?", options=["apple", "banana"])
        tok = _FakeTokenizer()
        scorer = _ScriptedScorer(
            tok,
            {
                # Run 1: 'apple' at A → model picks A with 0.9 (apple is right)
                "apple": [10.0, 7.8],
                # Run 2: 'banana' at A → model picks B with 0.9 (apple is still right, now at B)
                "banana": [7.8, 10.0],
            },
        )
        probs = debias(scorer, state_text="x", question=q, mode="reverse")
        # apple (canonical idx 0) got ~0.9 in both runs after mapping back
        assert probs[0] > 0.85
        assert probs[1] < 0.15


class TestDebiasModes:
    def test_none_calls_scorer_once(self) -> None:
        q = ChoiceQuestion(id="q", prompt="?", options=["x", "y", "z"])
        tok = _FakeTokenizer()
        scorer = _ScriptedScorer(tok, {"x": [1.0, 0.0, 0.0]})
        debias(scorer, state_text="s", question=q, mode="none")
        assert len(scorer.calls) == 1

    def test_cyclic_calls_scorer_k_times(self) -> None:
        q = ChoiceQuestion(id="q", prompt="?", options=["x", "y", "z"])
        tok = _FakeTokenizer()
        # Every permutation returns uniform logits — averaging stays uniform
        scorer = _ScriptedScorer(
            tok, {"x": [1.0, 1.0, 1.0], "y": [1.0, 1.0, 1.0], "z": [1.0, 1.0, 1.0]}
        )
        probs = debias(scorer, state_text="s", question=q, mode="cyclic")
        assert len(scorer.calls) == 3
        assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=0.01)

    def test_binary_short_circuits_reverse_to_single_pass(self) -> None:
        """BinaryQuestion + reverse should NOT run a second pass.

        Yes/No are semantic labels, not arbitrary positions. Reversing their
        order would compound the Yes-token prior rather than cancel position
        bias (see product/learnings/paws-below-random.md). The debias module
        short-circuits to a single-pass "none" for binary.
        """
        q = BinaryQuestion(id="q", prompt="urgent?")
        tok = _FakeTokenizer()
        scorer = _ScriptedScorer(tok, {"Yes": [10.0, 0.0], "No": [10.0, 0.0]})
        probs = debias(scorer, state_text="x", question=q, mode="reverse")

        # Exactly one score_batch call (one perm), not two.
        assert len(scorer.calls) == 1
        # Softmax of [10, 0] → [~1.0, ~0.0], so P(Yes) ≈ 1.0 (not the 0.5 that
        # the old buggy average produced).
        assert probs[0] > 0.99

    def test_binary_short_circuits_cyclic_too(self) -> None:
        """Same rationale — cyclic on binary is identical to reverse (one swap)."""
        q = BinaryQuestion(id="q", prompt="urgent?")
        tok = _FakeTokenizer()
        scorer = _ScriptedScorer(tok, {"Yes": [10.0, 0.0], "No": [10.0, 0.0]})
        _ = debias(scorer, state_text="x", question=q, mode="cyclic")
        assert len(scorer.calls) == 1  # single pass, not k=2

    def test_rating_question_runs_correct_number_of_permutations(self) -> None:
        q = RatingQuestion(id="q", prompt="?", scale=(1, 5))  # width 5
        tok = _FakeTokenizer()
        # Need 5 letter token ids to exist — _FakeTokenizer handles A..E via encode()
        mapping = letter_token_ids(tok)
        assert "E" in mapping
        scorer = _ScriptedScorer(
            tok,
            {str(v): [1.0] * 5 for v in range(1, 6)},
        )
        probs = debias(scorer, state_text="x", question=q, mode="cyclic")
        assert len(scorer.calls) == 5
        assert np.allclose(probs, [0.2] * 5, atol=0.01)
