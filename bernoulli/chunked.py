"""Chunked scoring for choice questions with more than 26 options.

The scorer's answer-position trick is capped at 26 label tokens (letters A..Z).
Banking77 and similar many-class problems blow past that. The vision doc §9
flags this as an open problem; we handle it by splitting the options into
chunks of <=26, scoring each chunk independently, and aggregating.

Aggregation strategy: keep **raw logits** per option and softmax once at the
end. Chunk-local softmaxes aren't comparable because they're normalized
within different label sets; raw logits at the ' A' answer position are
closer to a global confidence signal, albeit noisy (each chunk's prompt
differs because its labeled options differ).

Debiasing within chunks is deferred — mixing within-chunk prob averaging
with cross-chunk logit comparison is mathematically awkward. For now chunked
scoring is single-pass (no permutation). Measure first on Banking77, then
decide whether to add it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bernoulli.labels import LETTERS, letter_token_ids
from bernoulli.prompt import build_prompt
from bernoulli.scorer import Scorer
from bernoulli.types import ChoiceQuestion

CHUNK_SIZE = 26  # one letter per option within a chunk


def chunked_choice(
    scorer: Scorer,
    *,
    state_text: str,
    question: ChoiceQuestion,
    chunk_size: int = CHUNK_SIZE,
) -> NDArray[np.float32]:
    """Return per-option probabilities across all options after chunked scoring.

    Expects `question` to have more than `chunk_size` options; otherwise the
    standard single-pass path should be used.
    """
    tok = getattr(scorer, "tokenizer", None)
    if tok is None:
        raise RuntimeError("scorer has no .tokenizer attribute; needed for label token ids")

    options = question.options
    n = len(options)
    letter_ids = letter_token_ids(tok)

    # Build one prompt per chunk, batch them into a single scorer call.
    prompts: list[str] = []
    allowed_per_chunk: list[list[int]] = []
    starts: list[int] = []
    for start in range(0, n, chunk_size):
        chunk_opts = options[start : start + chunk_size]
        sub_question = ChoiceQuestion(id=question.id, prompt=question.prompt, options=chunk_opts)
        prompts.append(build_prompt(tok, state_text=state_text, question=sub_question))
        allowed_per_chunk.append([letter_ids[letter] for letter in LETTERS[: len(chunk_opts)]])
        starts.append(start)

    chunk_logits_list = scorer.score_batch(prompts, allowed_per_chunk)

    all_logits = np.zeros(n, dtype=np.float32)
    for start, chunk_logits in zip(starts, chunk_logits_list, strict=True):
        all_logits[start : start + len(chunk_logits)] = chunk_logits

    shifted = all_logits - all_logits.max()
    exp = np.exp(shifted)
    probs: NDArray[np.float32] = (exp / exp.sum()).astype(np.float32)
    return probs
