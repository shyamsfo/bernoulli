"""Pydantic types for the Bernoulli API (see vision_and_roadmap.md §3).

The API is three question kinds — choice, binary, rating — discriminated on the
`type` field. Each kind has its own response shape. All models are strict
(ConfigDict(extra='forbid')) so unknown keys surface as 422s rather than silent
guesses, per the vision doc.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State(BaseModel):
    """Unstructured state the model reads to answer the questions.

    One `state` is passed per request and shared across all questions in that
    request (the model reads it once; each question runs over the same state).
    Typical values: an email thread, a page of a terms-of-service document, a
    customer-support ticket, an earnings-release excerpt.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "text": "From: angry.customer@example.com\nSubject: STILL HAVEN'T SHIPPED\n...",
                "images": [],
            }
        },
    )

    text: str = Field(
        default="",
        description="State text, UTF-8. Everything the model should read before answering the questions.",
    )
    images: list[str] = Field(
        default_factory=list,
        description="Up to 8 images per state. Base64-encoded or local file paths. The scorer must support image inputs (VLMs; `scorer-hf` with a VLM backbone or `vllm`).",
        max_length=8,
    )


# ---------------------------------------------------------------------------
# Questions (discriminated union on `type`)
# ---------------------------------------------------------------------------


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Caller-supplied id, echoed as the key in `DecideResponse.decisions`.",
        examples=["intent", "chargeback_risk", "urgency"],
    )
    prompt: str = Field(
        description="The question text shown to the model. Decision-shaped (not open-ended).",
        examples=["What does the customer most want right now?"],
    )


class ChoiceQuestion(_QuestionBase):
    """Pick one of N string labels. The response includes the argmax label plus
    the full probability distribution across all options."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "intent",
                "type": "choice",
                "prompt": "What does the customer most want right now?",
                "options": ["refund", "ship_now", "status_update", "complaint_only", "other"],
            }
        },
    )

    type: Literal["choice"] = "choice"
    # Options above 26 (letter alphabet) are handled by chunked scoring in
    # bernoulli.chunked. Hard cap at 128 to keep context + compute bounded;
    # raise if a real need appears.
    options: list[str] = Field(
        min_length=2,
        max_length=128,
        description="At least 2, at most 128 string labels. Options beyond 26 are scored via chunking.",
    )


class BinaryQuestion(_QuestionBase):
    """Yes/no question. Response: a boolean + `probability` (P(answer=True))."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "chargeback_risk",
                "type": "binary",
                "prompt": "Is the customer threatening a chargeback or public complaint?",
            }
        },
    )

    type: Literal["binary"] = "binary"


class RatingQuestion(_QuestionBase):
    """Integer rating on an inclusive scale (e.g. 1–5). Response: an `expected`
    value under the full distribution across the rating labels."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "urgency",
                "type": "rating",
                "prompt": "How time-critical is this ticket? 1 = can wait a week, 5 = handle in the next hour.",
                "scale": [1, 5],
            }
        },
    )

    type: Literal["rating"] = "rating"
    scale: tuple[int, int] = Field(
        description="Inclusive [low, high] integer scale. Width (high - low + 1) must be ≤ 9.",
        examples=[[1, 5]],
    )

    @model_validator(mode="after")
    def _scale_valid(self) -> RatingQuestion:
        low, high = self.scale
        if high <= low:
            raise ValueError(f"rating scale high ({high}) must be > low ({low})")
        if high - low + 1 > 9:
            raise ValueError(
                f"rating scale width ({high - low + 1}) must fit single-digit labels (max 9)"
            )
        return self


Question = Annotated[
    ChoiceQuestion | BinaryQuestion | RatingQuestion,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Request options
# ---------------------------------------------------------------------------


class DecideOptions(BaseModel):
    """Per-request knobs. Sensible defaults are set; most callers don't need
    to override."""

    model_config = ConfigDict(extra="forbid")

    debias: Literal["none", "reverse", "cyclic"] = Field(
        default="reverse",
        description="Debiasing strategy to cancel label-position bias. `reverse` scores each permutation twice (default). `cyclic` is more thorough but costs ×N forward passes. `none` disables.",
    )
    calibrated: bool = Field(
        default=True,
        description="Apply the fitted temperature calibration if the server has one loaded (via `BERNOULLI_CALIBRATION_PATH`). Set to `false` to see raw softmax probabilities.",
    )


class DecideRequest(BaseModel):
    """The POST body for `/v1/decide` and `/v1/generate`."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "state": {
                    "text": "From: angry.customer@example.com\nSubject: STILL HAVEN'T SHIPPED\n\nThis is the third time I'm emailing...",
                },
                "questions": [
                    {
                        "id": "intent",
                        "type": "choice",
                        "prompt": "What does the customer most want right now?",
                        "options": ["refund", "ship_now", "status_update", "complaint_only", "other"],
                    },
                    {
                        "id": "urgency",
                        "type": "rating",
                        "prompt": "How time-critical is this ticket?",
                        "scale": [1, 5],
                    },
                    {
                        "id": "chargeback_risk",
                        "type": "binary",
                        "prompt": "Is the customer threatening a chargeback or public complaint?",
                    },
                ],
            }
        },
    )

    state: State = Field(description="Unstructured input the model reads once, shared across all questions.")
    questions: list[Question] = Field(
        min_length=1,
        description="One or more typed questions. Each gets a decision in the response, keyed by its `id`.",
    )
    options: DecideOptions = Field(
        default_factory=DecideOptions,
        description="Optional per-request knobs (debiasing, calibration). Defaults are production-shaped.",
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "Optional context shared across all questions in this request. "
            "Routed into the chat template's system turn, appended after the "
            "baseline single-letter instruction. Use for stable context "
            "(role description, rubric, policy) that doesn't vary with the "
            "state — e.g. a job description when the state is a resume. "
            "Note: the shipped calibration was fitted without instructions; "
            "passing a non-empty value may shift the model's prior enough "
            "that the fitted temperatures no longer apply."
        ),
    )


# ---------------------------------------------------------------------------
# Decisions (discriminated union, matched to question kind)
# ---------------------------------------------------------------------------


class _DecisionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceDecision(_DecisionBase):
    """Response variant for a `choice` question."""

    type: Literal["choice"] = "choice"
    answer: str = Field(description="The winning option string (argmax of `distribution`).")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of `answer`, i.e. `distribution[answer]`. Threshold on this.",
    )
    distribution: dict[str, float] = Field(
        description="Probability per option string. Sums to ~1.0.",
    )


class BinaryDecision(_DecisionBase):
    """Response variant for a `binary` question."""

    type: Literal["binary"] = "binary"
    answer: bool = Field(description="`True` iff `probability > 0.5`.")
    probability: float = Field(
        ge=0.0, le=1.0, description="P(answer=True). Threshold on this to branch."
    )


class RatingDecision(_DecisionBase):
    """Response variant for a `rating` question."""

    type: Literal["rating"] = "rating"
    expected: float = Field(
        description="Expected value under the full distribution: Σ i * P(rating=i). Use this when you want a scalar."
    )
    distribution: dict[str, float] = Field(
        description="Probability per integer label (as string). Sums to ~1.0."
    )


Decision = Annotated[
    ChoiceDecision | BinaryDecision | RatingDecision,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class DecideResponse(BaseModel):
    """The response body from `/v1/decide` and `/v1/generate`."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "decisions": {
                    "intent": {
                        "type": "choice",
                        "answer": "ship_now",
                        "confidence": 0.512,
                        "distribution": {
                            "refund": 0.458,
                            "ship_now": 0.512,
                            "status_update": 0.012,
                            "complaint_only": 0.004,
                            "other": 0.015,
                        },
                    },
                    "urgency": {
                        "type": "rating",
                        "expected": 4.93,
                        "distribution": {"1": 0.007, "2": 0.004, "3": 0.006, "4": 0.015, "5": 0.967},
                    },
                    "chargeback_risk": {
                        "type": "binary",
                        "answer": True,
                        "probability": 0.953,
                    },
                },
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "calibration_version": None,
                "latency_ms": 609,
            }
        },
    )

    decisions: dict[str, Decision] = Field(
        description="One `Decision` per question, keyed by `question.id`. The decision's `type` matches the question's `type`."
    )
    model: str = Field(description="The scorer's model id (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`).")
    calibration_version: str | None = Field(
        default=None,
        description="Identifier of the fitted calibration applied, if any. `null` means uncalibrated or `options.calibrated=false`.",
    )
    latency_ms: int = Field(description="Server-side wall-clock time for the forward passes, in milliseconds.")
