"""Shared baselines for every benchmark.

- **`Baseline` protocol** — the shape the runner consumes. Every baseline
  wraps the per-example call in `predict(example) -> Distribution`.
- **`BernoulliHTTP`** — POST to a running `/v1/decide`. The honest
  shipping path.
- **`Generative`** — POST to a running `/v1/generate`. Same backbone,
  prompted for text, parsed. Isolates the logit vs. generated-token
  calibration gap without a confounding model swap. HTTP so it shares
  the server's loaded scorer (no second GPU copy).
- **`DeBERTaZeroshot`** — `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
  via the HF `pipeline("zero-shot-classification", ...)`. Choice only.
- **`BGEm3LR`** — `BAAI/bge-m3` + a per-task scikit-learn LogisticRegression
  head. Needs a training split; the runner calls `.fit(train_examples)`
  before the predict loop.

The low-level generative primitives (`generative_decide`, `parse_response`)
live in `bernoulli.generative` so the server can import them without
`bernoulli/` reaching into `benchmarks/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from bernoulli.types import (
    BinaryDecision,
    ChoiceDecision,
    ChoiceQuestion,
    DecideResponse,
    Decision,
    Question,
    RatingDecision,
)

if TYPE_CHECKING:
    import httpx

    from benchmarks.common.dataset import BenchmarkExample


# ---------------------------------------------------------------------------
# Protocol + distribution flattening used by every wrapper.
# ---------------------------------------------------------------------------

Distribution = dict[str, float]


class Baseline(Protocol):
    """Common shape every baseline implements.

    `name` is the short, stable column id used in the leaderboard table
    (`bernoulli`, `generative`, `deberta`, `bge-m3-lr`). `predict` is
    called once per example; baselines own their own inference state
    (loaded model, HTTP client) and are constructed once per benchmark run.
    """

    name: str

    def predict(self, example: BenchmarkExample) -> Distribution: ...


def distribution_from_decision(decision: Decision, question: Question) -> Distribution:
    """Flatten a Decision into the {label_str: prob} shape metrics expects.

    - ChoiceDecision: already a distribution over option strings.
    - BinaryDecision: {'Yes': p, 'No': 1 - p}.
    - RatingDecision: already a distribution over integer-string keys.
    """
    if isinstance(decision, ChoiceDecision):
        return {k: float(v) for k, v in decision.distribution.items()}
    if isinstance(decision, BinaryDecision):
        return {"Yes": float(decision.probability), "No": float(1.0 - decision.probability)}
    if isinstance(decision, RatingDecision):
        return {k: float(v) for k, v in decision.distribution.items()}
    raise TypeError(f"unknown decision kind: {type(decision).__name__}")


# ---------------------------------------------------------------------------
# HTTP baselines: Bernoulli (/v1/decide) and Generative (/v1/generate).
# ---------------------------------------------------------------------------


class _HTTPDecider:
    """Shared behavior for POST-a-DecideRequest baselines.

    Subclass sets `name` + `_path` (`/v1/decide` or `/v1/generate`) and
    inherits the whole predict loop. The two baselines are 3 lines different;
    sharing avoids the drift risk of two near-copies.
    """

    name = "http-decider"
    _path = "/v1/decide"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        import httpx as _httpx

        self._client = client if client is not None else _httpx.Client(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    def predict(self, example: BenchmarkExample) -> Distribution:
        payload = {
            "state": {"text": example.state_text},
            "questions": [example.question.model_dump()],
        }
        r = self._client.post(f"{self._base_url}{self._path}", json=payload)
        r.raise_for_status()
        resp = DecideResponse.model_validate(r.json())
        return distribution_from_decision(resp.decisions[example.question.id], example.question)

    def close(self) -> None:
        self._client.close()


class BernoulliHTTP(_HTTPDecider):
    """Bernoulli over HTTP against a running `/v1/decide` server.

    One request per example — the server batches `questions x permutations`
    internally. Baselines are expected to be slower than production serving
    (per-example calls, no multi-example batching), which is honest for
    like-for-like comparison with the other baselines in this module.
    """

    name = "bernoulli"
    _path = "/v1/decide"


class Generative(_HTTPDecider):
    """Same-backbone generative baseline over `/v1/generate`.

    HTTP (not direct) so this baseline shares the server's already-loaded
    scorer — no second GPU copy on a single-GPU box. Fails fast if the
    server doesn't support generation (returns 501; raise_for_status
    surfaces it to the runner).
    """

    name = "generative"
    _path = "/v1/generate"


# ---------------------------------------------------------------------------
# DeBERTa-v3-zeroshot: strongest open encoder zero-shot classifier.
# "Did we need an LLM at all?" check for every academic benchmark.
# ---------------------------------------------------------------------------


class ZeroShotClassifier(Protocol):
    """Shape of the `transformers.pipeline('zero-shot-classification', ...)` callable.

    Factored into a Protocol so tests can inject a mock without pulling in
    an 800 MB HF model. Production code builds one via `DeBERTaZeroshot.load_default()`.
    """

    def __call__(
        self,
        sequences: str,
        candidate_labels: list[str],
        *,
        hypothesis_template: str = ...,
        multi_label: bool = ...,
    ) -> dict[str, Any]: ...


class DeBERTaZeroshot:
    """Zero-shot classifier via `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`.

    Choice questions map cleanly: the option strings become the candidate
    labels and the classifier's softmax over them is the output distribution.

    Binary and Rating questions raise NotImplementedError on purpose:

    - Binary "Yes"/"No" is semantic nonsense for a zero-shot classifier —
      the hypothesis template needs natural-language labels like
      "harmful"/"safe", which are per-benchmark. A benchmark that wants
      DeBERTa on a binary question should preprocess into a 2-way choice.
    - Rating over integer-string labels degrades to coin flips. Not worth
      reporting.

    The runner is expected to leave the DeBERTa column empty for benchmarks
    whose Question kind isn't Choice.
    """

    name = "deberta"

    _DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    _DEFAULT_TEMPLATE = "This text is about {}."

    def __init__(
        self,
        classifier: ZeroShotClassifier,
        *,
        hypothesis_template: str = _DEFAULT_TEMPLATE,
    ) -> None:
        self._classifier = classifier
        self._hypothesis_template = hypothesis_template

    @classmethod
    def load_default(
        cls,
        *,
        model_id: str = _DEFAULT_MODEL,
        device: int | str = -1,
        hypothesis_template: str = _DEFAULT_TEMPLATE,
    ) -> DeBERTaZeroshot:
        """Build with the default HF zero-shot pipeline on `device` (-1 for CPU, 0 for cuda:0)."""
        from transformers import pipeline

        clf = pipeline("zero-shot-classification", model=model_id, device=device)
        return cls(cast(ZeroShotClassifier, clf), hypothesis_template=hypothesis_template)

    def predict(self, example: BenchmarkExample) -> Distribution:
        if not isinstance(example.question, ChoiceQuestion):
            raise NotImplementedError(
                f"DeBERTaZeroshot supports ChoiceQuestion only; got "
                f"{type(example.question).__name__}. Skip this baseline column "
                f"on binary/rating benchmarks."
            )
        options = list(example.question.options)
        result = self._classifier(
            example.state_text,
            candidate_labels=options,
            hypothesis_template=self._hypothesis_template,
            multi_label=False,
        )
        # HF pipeline returns labels in confidence-sorted order; realign to options.
        labels = cast(list[str], result["labels"])
        scores = cast(list[float], result["scores"])
        by_label = dict(zip(labels, scores, strict=True))
        return {opt: float(by_label[opt]) for opt in options}


# ---------------------------------------------------------------------------
# BGE-m3 + per-task logistic regression: "could we have used embeddings?" check.
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Shape of the BGE-m3 encode call. Factored out so tests inject a fake."""

    def __call__(self, texts: list[str]) -> NDArray[np.float32]: ...


class BGEm3LR:
    """BGE-m3 sentence embedder + per-task scikit-learn LogisticRegression.

    Unlike the other baselines, this one needs training data. Call
    `.fit(train_examples)` once per benchmark before `.predict()`. The
    runner handles this with a `hasattr(baseline, "fit")` branch —
    deliberately not on the base `Baseline` protocol, which stays
    predict-only.

    Supports Choice, Binary, and Rating uniformly — scikit-learn's
    LogisticRegression handles string labels of any arity. Rating labels
    lose their ordinal semantics (treated as nominal classes) but still
    produce a usable distribution over the integer-string keys.

    All examples in a `fit()` call must share the same question kind
    AND label space (every benchmark trains its own head on its own
    labels); the runner enforces this by scoping fit per benchmark.
    """

    name = "bge-m3-lr"

    _DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        embed_fn: Embedder,
        *,
        classifier: Any | None = None,
    ) -> None:
        self._embed = embed_fn
        self._clf = classifier  # fitted sklearn LogisticRegression after .fit()
        self._classes: list[str] | None = None

    @classmethod
    def load_default(
        cls,
        *,
        model_id: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> BGEm3LR:
        """Build with a transformers-backed BGE-m3 embedder on `device` (`cpu` or `cuda`)."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()

        @torch.no_grad()
        def embed(texts: list[str]) -> NDArray[np.float32]:
            vectors: list[NDArray[np.float32]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc)
                cls_vec = out.last_hidden_state[:, 0]
                # BGE convention: L2-normalize the CLS vector.
                cls_vec = cls_vec / cls_vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                vectors.append(cls_vec.cpu().to(torch.float32).numpy())
            return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), np.float32)

        return cls(embed)

    def fit(self, train_examples: list[BenchmarkExample]) -> None:
        """Train the LR head on `train_examples`. Idempotent — a second fit replaces the head."""
        from sklearn.linear_model import LogisticRegression

        if not train_examples:
            raise ValueError("BGEm3LR.fit requires at least one training example")
        texts = [ex.state_text for ex in train_examples]
        labels = [ex.gold for ex in train_examples]
        features = self._embed(texts)
        self._clf = LogisticRegression(max_iter=1000).fit(features, labels)
        # sklearn exposes classes_ in sorted order after fit; preserve that order.
        self._classes = list(map(str, self._clf.classes_))

    def predict(self, example: BenchmarkExample) -> Distribution:
        if self._clf is None or self._classes is None:
            raise RuntimeError(
                "BGEm3LR.predict called before .fit(). The runner must call fit "
                "with the benchmark's labeled train split before scoring."
            )
        features = self._embed([example.state_text])
        probs = self._clf.predict_proba(features)[0]
        return {label: float(p) for label, p in zip(self._classes, probs, strict=True)}
