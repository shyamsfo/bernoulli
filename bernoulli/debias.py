"""Position/label debiasing.

The model has a position bias — all else equal it slightly prefers whichever
label appears first. The vision doc §4 and the XOR paper handle this by
rerunning the same request with permuted option orders and averaging the
probabilities after mapping back to the canonical option index.

Modes:
    none    — no reruns; cheap passthrough
    reverse — one extra forward pass with options in reversed order (default)
    cyclic  — k forward passes, one per cyclic shift of k options; k <= 6
              because 7+ passes rarely beats reverse and the cost adds up fast

API: `debias(scorer, state_text, question, mode)` returns a probability
array indexed by canonical option order.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from bernoulli.labels import binary_token_ids, letter_token_ids
from bernoulli.prompt import build_prompt, labels_for
from bernoulli.scorer import Scorer
from bernoulli.types import BinaryQuestion, Question

CYCLIC_MAX_OPTIONS = 6
DebiasMode = Literal["none", "reverse", "cyclic"]


def _permutations_for(mode: DebiasMode, n: int) -> list[list[int]]:
    """Return the list of option_order permutations for this mode."""
    if mode == "none":
        return [list(range(n))]
    if mode == "reverse":
        if n <= 1:
            return [list(range(n))]
        return [list(range(n)), list(reversed(range(n)))]
    if mode == "cyclic":
        if n > CYCLIC_MAX_OPTIONS:
            raise ValueError(
                f"cyclic debias is capped at k <= {CYCLIC_MAX_OPTIONS} options; got {n}"
            )
        return [[(i + shift) % n for i in range(n)] for shift in range(n)]
    raise ValueError(f"unknown debias mode: {mode!r}")


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    result: NDArray[np.float32] = exp / exp.sum()
    return result


def _allowed_ids(tok: object, question: Question) -> list[int]:
    """Token ids for the question's label set, in canonical label order."""
    labels = labels_for(question)
    if isinstance(question, BinaryQuestion):
        mapping = binary_token_ids(tok)  # type: ignore[arg-type]
    else:
        mapping = letter_token_ids(tok)  # type: ignore[arg-type]
    return [mapping[label] for label in labels]


def debias(
    scorer: Scorer,
    *,
    state_text: str,
    question: Question,
    mode: DebiasMode = "reverse",
) -> NDArray[np.float32]:
    """Return per-option probabilities in canonical order after averaging runs.

    One `scorer.score` call per permutation. For mode='reverse' that's 2;
    for 'cyclic' k <= 6 it's k. 'none' is a single call (passthrough).
    """
    tok = getattr(scorer, "tokenizer", None)
    if tok is None:
        raise RuntimeError("scorer has no .tokenizer attribute; needed to resolve label token ids")

    allowed = _allowed_ids(tok, question)
    n = len(allowed)
    perms = _permutations_for(mode, n)

    # Build all permutation prompts up front, send as a single batch — one
    # engine call on vLLM's continuous-batching path, loop on HFScorer.
    prompts = [
        build_prompt(tok, state_text=state_text, question=question, option_order=perm)
        for perm in perms
    ]
    all_logits = scorer.score_batch(prompts, [allowed] * len(perms))

    totals = np.zeros(n, dtype=np.float32)
    for perm, logits in zip(perms, all_logits, strict=True):
        probs = _softmax(logits)
        # probs[i] is the model's probability for the option at label position i,
        # which corresponds to canonical option `perm[i]`. Map back.
        for i, canon_idx in enumerate(perm):
            totals[canon_idx] += probs[i]

    averaged: NDArray[np.float32] = (totals / len(perms)).astype(np.float32)
    return averaged
