"""FastAPI server exposing /v1/decide, /v1/generate, /healthz, /v1/models.

Thin HTTP wrapper around `bernoulli.decide.decide()` + `bernoulli.generative.
generative_decide()`. Loads the scorer and (optional) calibration once at
startup via a lifespan context, then serves requests off a thread pool so
blocking GPU work doesn't stall the event loop.

`/v1/generate` is the baseline/comparison path: same request shape as
/v1/decide, but internally produces its distribution by generating a short
completion and parsing it (1.0/0.0 point mass on the parsed answer, uniform
on parse failure). Only supported when the loaded scorer has a `.generate()`
method — HFScorer does, VLLMScorer does not; a request against a non-
generating scorer gets 501.

Run with:
    uvicorn bernoulli.server:app --host 0.0.0.0 --port 8000

Config: everything comes from `BERNOULLI_*` env vars (see bernoulli.config).
Calibration path is `BERNOULLI_CALIBRATION_PATH` — unset means no calibration
is applied even if request.options.calibrated=True.

Request batching across questions is deferred to M4c (needs scorer-level
support in VLLMScorer); this M4b server runs them sequentially inside one
decide() call.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, HTTPException

from bernoulli.calibrate import Calibration, load_calibration
from bernoulli.config import load_settings
from bernoulli.decide import decide
from bernoulli.generative import generative_decide
from bernoulli.scorer import Scorer, load_scorer
from bernoulli.types import DecideRequest, DecideResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _load_calibration_if_configured() -> Calibration | None:
    path = os.environ.get("BERNOULLI_CALIBRATION_PATH")
    if not path:
        return None
    return load_calibration(Path(path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the scorer + optional calibration before the server accepts traffic."""
    settings = load_settings()
    app.state.settings = settings
    app.state.scorer = load_scorer(settings)
    app.state.calibration = _load_calibration_if_configured()
    yield


def create_app(
    *,
    scorer: Scorer | None = None,
    calibration: Calibration | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    If `scorer` is provided, skip the lifespan model-load and inject it
    directly (used by tests to pass a mock scorer; production call-sites
    omit this and let lifespan handle it).
    """
    if scorer is not None:
        app = FastAPI(title="Bernoulli", version="0.0.1")
        app.state.scorer = scorer
        app.state.calibration = calibration
        app.state.settings = None
    else:
        app = FastAPI(title="Bernoulli", version="0.0.1", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, list[dict[str, object]]]:
        scorer_: Scorer = app.state.scorer
        return {
            "models": [
                {
                    "id": scorer_.model_id,
                    "revision": scorer_.revision,
                }
            ]
        }

    @app.post("/v1/decide", response_model=DecideResponse)
    def decide_endpoint(request: DecideRequest) -> DecideResponse:
        try:
            return decide(
                request,
                cast(Scorer, app.state.scorer),
                calibration=app.state.calibration,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/v1/generate", response_model=DecideResponse)
    def generate_endpoint(request: DecideRequest) -> DecideResponse:
        scorer_ = app.state.scorer
        if not hasattr(scorer_, "generate"):
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Loaded scorer ({type(scorer_).__name__}) has no .generate(); "
                    f"use BERNOULLI_SCORER=hf for the generative baseline path."
                ),
            )
        try:
            return generative_decide(request, scorer_)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    return app


app = create_app()
