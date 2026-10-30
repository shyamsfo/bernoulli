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
    """Unstructured state: text + optional images (base64 or file paths)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="", description="State text content")
    images: list[str] = Field(
        default_factory=list,
        description="Up to 8 images per state. Base64-encoded or local file paths.",
        max_length=8,
    )


# ---------------------------------------------------------------------------
# Questions (discriminated union on `type`)
# ---------------------------------------------------------------------------


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Caller-supplied id, echoed in the response")
    prompt: str = Field(description="The question text shown to the model")


class ChoiceQuestion(_QuestionBase):
    type: Literal["choice"] = "choice"
    # Options above 26 (letter alphabet) are handled by chunked scoring in
    # bernoulli.chunked. Hard cap at 128 to keep context + compute bounded;
    # raise if a real need appears.
    options: list[str] = Field(min_length=2, max_length=128)


class BinaryQuestion(_QuestionBase):
    type: Literal["binary"] = "binary"


class RatingQuestion(_QuestionBase):
    type: Literal["rating"] = "rating"
    scale: tuple[int, int] = Field(description="Inclusive [low, high] integer scale")

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
    model_config = ConfigDict(extra="forbid")

    debias: Literal["none", "reverse", "cyclic"] = "reverse"
    calibrated: bool = True


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: State
    questions: list[Question] = Field(min_length=1)
    options: DecideOptions = Field(default_factory=DecideOptions)


# ---------------------------------------------------------------------------
# Decisions (discriminated union, matched to question kind)
# ---------------------------------------------------------------------------


class _DecisionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceDecision(_DecisionBase):
    type: Literal["choice"] = "choice"
    answer: str = Field(description="The winning option string (argmax of distribution)")
    confidence: float = Field(ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(description="Probability per option string")


class BinaryDecision(_DecisionBase):
    type: Literal["binary"] = "binary"
    answer: bool
    probability: float = Field(ge=0.0, le=1.0, description="P(answer=True)")


class RatingDecision(_DecisionBase):
    type: Literal["rating"] = "rating"
    expected: float = Field(description="E[rating] under the distribution")
    distribution: dict[str, float] = Field(description="Probability per integer label (as str)")


Decision = Annotated[
    ChoiceDecision | BinaryDecision | RatingDecision,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class DecideResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: dict[str, Decision]
    model: str
    calibration_version: str | None = None
    latency_ms: int
