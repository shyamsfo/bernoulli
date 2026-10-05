"""Chunked choice scoring tests (mock scorer, no GPU)."""

from __future__ import annotations

import re

import numpy as np

from bernoulli.chunked import CHUNK_SIZE, chunked_choice
from bernoulli.labels import letter_token_ids
from bernoulli.types import ChoiceQuestion


class _FakeTokenizer:
    """Letter-token + chat-template mock sufficient for chunked_choice()."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text.startswith(" ") and len(text) == 2 and text[1].isalpha():
            return [1000 + ord(text[1])]
        if text == " Yes":
            return [1200]
        if text == " No":
            return [1201]
        return [0, 0]

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> str:
        return str(next(m["content"] for m in conversation if m["role"] == "user"))


class _LogitsByFirstOption:
    """Scorer that returns scripted logits keyed by the first option rendered.

    This lets a test control what each chunk's forward pass sees. The logits
    returned are RAW (chunked_choice is responsible for softmaxing globally).
    """

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
        match = re.search(r"\n([A-Za-z]+)\) ([^\n]+)", prompt)
        if match is None:
            raise AssertionError(f"could not find first option in prompt:\n{prompt}")
        first = match.group(2).strip()
        if first not in self._logits_by_first_option:
            raise AssertionError(
                f"no scripted logits for first option {first!r}; "
                f"keys={list(self._logits_by_first_option)}"
            )
        return np.asarray(self._logits_by_first_option[first], dtype=np.float32)

    def score_batch(
        self,
        prompts: list[str],
        allowed_token_ids_list: list[list[int]],
        *,
        images_list: list[list[str] | None] | None = None,
    ) -> list[np.ndarray]:
        return [self.score(p, a) for p, a in zip(prompts, allowed_token_ids_list, strict=True)]


def _options(n: int) -> list[str]:
    return [f"opt{i:02d}" for i in range(n)]


class TestChunking:
    def test_three_chunks_for_banking77_sized(self) -> None:
        """77 options should split into 3 chunks of sizes 26, 26, 25."""
        tok = _FakeTokenizer()
        letter_tokens = letter_token_ids(tok)  # verifies the fake tokenizer works
        assert len(letter_tokens) == 26

        n = 77
        options = _options(n)
        q = ChoiceQuestion(id="q", prompt="?", options=options)

        # Scripted logits: give chunk-0 first option a strong 10.0, others 0.0.
        # chunk 0 starts with opt00, chunk 1 with opt26, chunk 2 with opt52.
        logits_table = {
            "opt00": [10.0] + [0.0] * 25,  # 26 options
            "opt26": [0.0] * 26,
            "opt52": [0.0] * 25,  # 25 options
        }
        scorer = _LogitsByFirstOption(tok, logits_table)
        probs = chunked_choice(scorer, state_text="x", question=q)

        assert len(scorer.calls) == 3  # one forward pass per chunk
        assert probs.shape == (77,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
        # opt00 had the only non-zero logit → it should dominate
        assert probs.argmax() == 0

    def test_cross_chunk_logit_comparison(self) -> None:
        """A high raw logit in a later chunk should beat a lower logit in chunk 0."""
        tok = _FakeTokenizer()
        n = 52  # exactly 2 chunks of 26
        q = ChoiceQuestion(id="q", prompt="?", options=_options(n))

        logits_table = {
            "opt00": [5.0] + [0.0] * 25,  # chunk 0: opt00 is a modest 5.0
            "opt26": [0.0] * 10 + [15.0] + [0.0] * 15,  # chunk 1: opt36 is a strong 15.0
        }
        scorer = _LogitsByFirstOption(tok, logits_table)
        probs = chunked_choice(scorer, state_text="x", question=q)

        # Global argmax should be opt36 (idx 36), not opt00 (idx 0)
        assert probs.argmax() == 36
        # And its probability should dominate
        assert probs[36] > 0.9

    def test_single_chunk_when_within_cap(self) -> None:
        """Even when options fit in one chunk, chunked_choice should work."""
        tok = _FakeTokenizer()
        q = ChoiceQuestion(id="q", prompt="?", options=_options(5))
        logits_table = {"opt00": [10.0, 0.0, 0.0, 0.0, 0.0]}
        scorer = _LogitsByFirstOption(tok, logits_table)
        probs = chunked_choice(scorer, state_text="x", question=q)
        assert len(scorer.calls) == 1
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
        assert probs.argmax() == 0


class TestChunkSizeConstant:
    def test_matches_letter_alphabet(self) -> None:
        assert CHUNK_SIZE == 26
