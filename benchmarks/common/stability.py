"""Reorder + reword stability runner.

For each example in a benchmark:

- **Reorder**: regenerate the `ChoiceQuestion` with the options in a
  permuted order and ask the baseline again. For `N <= 6` options we use
  every non-identity permutation; for `N > 6` we sample
  `max_reorder_perms` permutations with a seeded RNG. BinaryQuestion and
  RatingQuestion are skipped — binary has no meaningful "reorder" ("Yes"
  vs "No" is semantic, not positional) and rating options are
  numerically ordered.
- **Reword**: swap the question's prompt for each of the (up to three)
  hand-authored paraphrases in `benchmarks/academic/<name>/loader.py ::
  REWORD_STEMS`. Options stay the same.

An example is **stable under reorder / reword** iff its top-1 label is
unchanged across every variant for that mode. The two scores are
reported independently.

The runner is baseline-agnostic — any `Baseline` works. It calls
`.predict(example)` once per variant and uses the metrics module's
`stability()` helper for the final aggregation.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.common.baselines import Baseline, Distribution
from benchmarks.common.dataset import BenchmarkExample
from benchmarks.common.metrics import stability
from bernoulli.types import ChoiceQuestion


def _top1(dist: Distribution) -> str:
    """Return the argmax key — ties broken by (negative) value then key."""
    return max(dist.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _reorder_variants(options: list[str], max_perms: int, rng: random.Random) -> list[list[str]]:
    """Return option-order permutations to test, excluding the identity.

    - `len(options) <= 6` → every non-identity permutation (up to 719).
    - otherwise → `max_perms` sampled permutations with the given RNG.
    """
    identity = tuple(options)
    if len(options) <= 6:
        all_perms = [list(p) for p in itertools.permutations(options) if p != identity]
        return all_perms
    variants: list[list[str]] = []
    seen: set[tuple[str, ...]] = {identity}
    while len(variants) < max_perms:
        shuffled = list(options)
        rng.shuffle(shuffled)
        key = tuple(shuffled)
        if key in seen:
            continue
        seen.add(key)
        variants.append(shuffled)
    return variants


def _with_options(question: ChoiceQuestion, options: list[str]) -> ChoiceQuestion:
    return ChoiceQuestion(id=question.id, prompt=question.prompt, options=options)


def _with_prompt(question: ChoiceQuestion, prompt: str) -> ChoiceQuestion:
    return ChoiceQuestion(id=question.id, prompt=prompt, options=list(question.options))


@dataclass(frozen=True)
class StabilityResult:
    """Headline numbers from a stability run.

    - `reorder_score`: fraction of **choice** examples whose top-1 is
      unchanged across every reorder variant. `None` if the benchmark has
      no `ChoiceQuestion` examples.
    - `reword_score`: fraction of **all** examples whose top-1 is
      unchanged across every reword variant.
    - `n_reorder_examples`, `n_reword_examples`: counts the scores were
      computed over, surfaced for honest reporting.
    - `reorder_variants_per_example`: how many permutations each choice
      example was tested against.
    """

    reorder_score: float | None
    reword_score: float
    n_reorder_examples: int
    n_reword_examples: int
    reorder_variants_per_example: int


def stability_run(
    baseline: Baseline,
    examples: Sequence[BenchmarkExample],
    reword_stems: Sequence[str],
    *,
    max_reorder_perms: int = 20,
    seed: int = 0,
) -> StabilityResult:
    """Run reorder + reword stability for one baseline + one benchmark's examples.

    `examples` is materialised (not an iterator) because every example is
    probed multiple times. The caller is expected to apply any `limit`
    before passing in — a full benchmark x full stability sweep can be
    `O(N x (perms + rewords))` baseline calls.
    """
    rng = random.Random(seed)

    reference: list[str] = [_top1(baseline.predict(ex)) for ex in examples]

    # --- reorder ---------------------------------------------------------
    choice_mask = [isinstance(ex.question, ChoiceQuestion) for ex in examples]
    choice_examples = [ex for ex, is_choice in zip(examples, choice_mask, strict=True) if is_choice]
    choice_refs = [ref for ref, is_choice in zip(reference, choice_mask, strict=True) if is_choice]

    reorder_score: float | None = None
    reorder_perms_per_example = 0
    if choice_examples:
        # Permutation set is a function of (option tuple, max_perms) — most
        # benchmarks have a single ChoiceQuestion, so this is one shuffle.
        # Still, compute per-example to be safe when a benchmark varies options.
        variant_runs: list[list[str]] = []
        for i, ex in enumerate(choice_examples):
            question = ex.question
            assert isinstance(question, ChoiceQuestion)
            variants = _reorder_variants(list(question.options), max_reorder_perms, rng)
            if i == 0:
                reorder_perms_per_example = len(variants)
                variant_runs = [[] for _ in variants]
            for j, variant_options in enumerate(variants):
                variant_ex = BenchmarkExample(
                    state_text=ex.state_text,
                    question=_with_options(question, variant_options),
                    gold=ex.gold,
                    source=ex.source,
                    meta=ex.meta,
                )
                variant_runs[j].append(_top1(baseline.predict(variant_ex)))
        reorder_score = stability(choice_refs, variant_runs)

    # --- reword ----------------------------------------------------------
    reword_runs: list[list[str]] = [[] for _ in reword_stems]
    for ex in examples:
        for j, stem in enumerate(reword_stems):
            # Reword only wraps the question's prompt — type-dependent.
            new_question: object
            if isinstance(ex.question, ChoiceQuestion):
                new_question = _with_prompt(ex.question, stem)
            else:
                # Binary / Rating / future kinds: rebuild via model_copy to flip prompt only.
                new_question = ex.question.model_copy(update={"prompt": stem})
            variant_ex = BenchmarkExample(
                state_text=ex.state_text,
                question=new_question,  # type: ignore[arg-type]
                gold=ex.gold,
                source=ex.source,
                meta=ex.meta,
            )
            reword_runs[j].append(_top1(baseline.predict(variant_ex)))
    reword_score = stability(reference, reword_runs) if reword_stems else 1.0

    return StabilityResult(
        reorder_score=reorder_score,
        reword_score=reword_score,
        n_reorder_examples=len(choice_examples),
        n_reword_examples=len(examples),
        reorder_variants_per_example=reorder_perms_per_example,
    )
