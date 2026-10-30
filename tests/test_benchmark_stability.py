"""Unit tests for benchmarks/common/stability.py.

A scripted baseline makes the semantics deterministic — tests cover the
reorder-only path, the reword-only path, both together, mixed-question-type
benchmarks (reorder applies only to Choice), and the N <= 6 full-permutation
vs N > 6 sampled branch.
"""

from __future__ import annotations

from benchmarks.common.baselines import Distribution
from benchmarks.common.dataset import BenchmarkExample
from benchmarks.common.stability import StabilityResult, stability_run
from bernoulli.types import BinaryQuestion, ChoiceQuestion, RatingQuestion


class _ScriptedBaseline:
    """Baseline whose output is a function of (state_text, prompt, options_tuple).

    Each test configures a predict_fn that returns a Distribution; the
    runner's bookkeeping is what's under test, not probability math.
    """

    name = "scripted"

    def __init__(self, predict_fn: object) -> None:
        self._fn = predict_fn

    def predict(self, example: BenchmarkExample) -> Distribution:
        options = _options_tuple(example)
        prompt = example.question.prompt
        return self._fn(example.state_text, prompt, options)  # type: ignore[operator]


def _options_tuple(ex: BenchmarkExample) -> tuple[str, ...]:
    q = ex.question
    if isinstance(q, ChoiceQuestion):
        return tuple(q.options)
    if isinstance(q, BinaryQuestion):
        return ("Yes", "No")
    if isinstance(q, RatingQuestion):
        low, high = q.scale
        return tuple(str(i) for i in range(low, high + 1))
    raise TypeError(f"unknown question kind: {type(q).__name__}")


def _choice_example(state: str, options: list[str], gold: str) -> BenchmarkExample:
    q = ChoiceQuestion(id="x", prompt="pick one", options=options)
    return BenchmarkExample(state_text=state, question=q, gold=gold, source="t")


def _binary_example(state: str, gold: str) -> BenchmarkExample:
    q = BinaryQuestion(id="x", prompt="yes or no?")
    return BenchmarkExample(state_text=state, question=q, gold=gold, source="t")


class TestReorderStability:
    def test_perfectly_stable_classifier_is_fully_stable(self) -> None:
        # Always predict "a" regardless of prompt/options.
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {opt: (0.9 if opt == "a" else 0.1 / (len(options) - 1)) for opt in options}

        examples = [_choice_example("hi", ["a", "b", "c"], gold="a") for _ in range(3)]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=[])
        assert result.reorder_score == 1.0
        assert result.n_reorder_examples == 3
        # 3! - 1 = 5 variants (identity excluded)
        assert result.reorder_variants_per_example == 5

    def test_position_sensitive_classifier_is_fully_unstable(self) -> None:
        # Always predict the first option — top-1 moves with reorder.
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {opt: (1.0 if i == 0 else 0.0) for i, opt in enumerate(options)}

        examples = [_choice_example("hi", ["a", "b", "c"], gold="a") for _ in range(4)]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=[])
        assert result.reorder_score == 0.0
        assert result.n_reorder_examples == 4

    def test_large_option_set_uses_sampled_permutations(self) -> None:
        options = [chr(ord("a") + i) for i in range(10)]  # 10 options → sampled path

        def predict(state: str, prompt: str, opts: tuple[str, ...]) -> Distribution:
            return {opt: (1.0 if opt == "a" else 0.0) for opt in opts}

        examples = [_choice_example("x", options, gold="a")]
        result = stability_run(
            _ScriptedBaseline(predict), examples, reword_stems=[], max_reorder_perms=7, seed=42
        )
        assert result.reorder_score == 1.0
        assert result.reorder_variants_per_example == 7


class TestRewordStability:
    def test_prompt_invariant_classifier_is_stable(self) -> None:
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {opt: (0.9 if opt == "yes" else 0.1) for opt in options}

        q = ChoiceQuestion(id="x", prompt="orig", options=["yes", "no"])
        examples = [
            BenchmarkExample(state_text=f"s{i}", question=q, gold="yes", source="t")
            for i in range(5)
        ]
        rewords = ["phrasing A?", "phrasing B?", "phrasing C?"]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=rewords)
        assert result.reword_score == 1.0
        assert result.n_reword_examples == 5

    def test_prompt_dependent_classifier_can_be_unstable(self) -> None:
        # If prompt mentions "flip", invert the prediction.
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            if "flip" in prompt:
                return {opt: (0.9 if opt == "no" else 0.1) for opt in options}
            return {opt: (0.9 if opt == "yes" else 0.1) for opt in options}

        q = ChoiceQuestion(id="x", prompt="original", options=["yes", "no"])
        examples = [BenchmarkExample(state_text="s", question=q, gold="yes", source="t")]
        # One reword flips, one doesn't → example not stable (needs ALL to agree).
        rewords = ["same wording", "please flip the answer", "and once more"]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=rewords)
        assert result.reword_score == 0.0

    def test_no_rewords_counts_as_fully_stable(self) -> None:
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {opt: 1.0 / len(options) for opt in options}

        examples = [_choice_example("s", ["a", "b"], gold="a")]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=[])
        # No rewords ⇒ nothing to disagree with; metrics.stability returns 1.0.
        assert result.reword_score == 1.0
        assert result.n_reword_examples == 1


class TestMixedQuestionKinds:
    def test_reorder_skipped_on_binary_only_benchmark(self) -> None:
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {"Yes": 0.8, "No": 0.2}

        examples = [_binary_example("s1", gold="Yes"), _binary_example("s2", gold="Yes")]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=["again?"])
        assert result.reorder_score is None
        assert result.n_reorder_examples == 0
        assert result.reword_score == 1.0

    def test_reword_covers_all_kinds(self) -> None:
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {opt: (1.0 if i == 0 else 0.0) for i, opt in enumerate(options)}

        q_rating = RatingQuestion(id="r", prompt="rate", scale=(1, 3))
        examples = [
            _choice_example("s1", ["a", "b"], gold="a"),
            _binary_example("s2", gold="Yes"),
            BenchmarkExample(state_text="s3", question=q_rating, gold="1", source="t"),
        ]
        rewords = ["phrased differently"]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=rewords)
        assert result.n_reword_examples == 3
        assert result.reword_score == 1.0


class TestStabilityResult:
    def test_default_shape(self) -> None:
        def predict(state: str, prompt: str, options: tuple[str, ...]) -> Distribution:
            return {opt: 1.0 / len(options) for opt in options}

        examples = [_choice_example("s", ["a", "b"], gold="a")]
        result = stability_run(_ScriptedBaseline(predict), examples, reword_stems=["r1"])
        assert isinstance(result, StabilityResult)
        assert 0.0 <= result.reword_score <= 1.0
        assert result.reorder_score is not None
        assert 0.0 <= result.reorder_score <= 1.0
