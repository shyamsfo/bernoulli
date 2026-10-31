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

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from bernoulli.calibrate import Calibration, load_calibration
from bernoulli.config import load_settings
from bernoulli.decide import decide
from bernoulli.generative import generative_decide
from bernoulli.scorer import Scorer, load_scorer
from bernoulli.types import DecideRequest, DecideResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


def _load_calibration_if_configured() -> Calibration | None:
    path = os.environ.get("BERNOULLI_CALIBRATION_PATH")
    if not path:
        return None
    return load_calibration(Path(path))


def _load_api_keys() -> frozenset[str]:
    """Parse BERNOULLI_API_KEYS as comma-separated tokens. Empty set = auth disabled."""
    raw = os.environ.get("BERNOULLI_API_KEYS", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


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
    api_keys: frozenset[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    If `scorer` is provided, skip the lifespan model-load and inject it
    directly (used by tests to pass a mock scorer; production call-sites
    omit this and let lifespan handle it).

    `api_keys`: when non-empty, every request except /healthz must carry
    `Authorization: Bearer <key>` where <key> is one of the configured
    tokens. If None, keys are read from the BERNOULLI_API_KEYS env var at
    construction time (comma-separated). An empty set disables auth —
    appropriate for local dev and tests, never for a public endpoint.
    """
    if api_keys is None:
        api_keys = _load_api_keys()

    description = (
        "Bernoulli is an open-weight decision model. POST unstructured state "
        "(text, optional images) and a lens pack (one or more typed questions) "
        "to `/v1/decide`; get back a calibrated probability distribution per "
        "question in a single forward pass over the state.\n\n"
        "**Three question types**, discriminated on the `type` field:\n"
        "- `choice` — pick one of N labels\n"
        "- `binary` — yes/no\n"
        "- `rating` — integer scale (e.g. 1–5)\n\n"
        "**Auth** (optional, server-configured): `Authorization: Bearer <api-key>`. "
        "Set `BERNOULLI_API_KEYS` on the server to a comma-separated list of "
        "accepted tokens; an empty list disables auth. `/healthz` and these "
        "docs endpoints (`/docs`, `/redoc`, `/openapi.json`) are always open.\n\n"
        "Project: [github.com/shyamsfo/bernoulli](https://github.com/shyamsfo/bernoulli) · "
        "Site: [bernoulli.live](https://www.bernoulli.live/)"
    )
    app_kwargs = {
        "title": "Bernoulli — decision API",
        "version": "0.0.1",
        "description": description,
        "contact": {
            "name": "DeepStore",
            "email": "contact@deepstore.ai",
            "url": "https://www.bernoulli.live/",
        },
        "license_info": {
            "name": "Apache 2.0",
            "url": "https://github.com/shyamsfo/bernoulli/blob/main/LICENSE",
        },
        "openapi_tags": [
            {"name": "decide", "description": "Score decisions over state."},
            {"name": "introspection", "description": "What's loaded on the server."},
            {"name": "ops", "description": "Health + liveness probes."},
        ],
    }
    if scorer is not None:
        app = FastAPI(**app_kwargs)
        app.state.scorer = scorer
        app.state.calibration = calibration
        app.state.settings = None
    else:
        app = FastAPI(**app_kwargs, lifespan=lifespan)

    @app.middleware("http")
    async def _api_key_auth(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # /healthz is intentionally unauthenticated so load balancers,
        # uptime monitors, and the serve-up wait-loop can probe without
        # a key. The docs endpoints (/docs, /redoc, /openapi.json) are
        # also open — they're a product surface (prospective callers read
        # them before signing up); the actual data endpoints below stay
        # gated.
        if request.url.path in ("/healthz", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        # No keys configured = auth disabled. Dev / test only.
        if not api_keys:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "missing bearer token"})
        token = auth[len("Bearer ") :]
        if token not in api_keys:
            return JSONResponse(status_code=401, content={"detail": "invalid api key"})
        return await call_next(request)

    @app.get("/healthz", tags=["ops"], summary="Health check")
    def healthz() -> dict[str, str]:
        """200 with `{"status": "ok"}` once the server is bound and the scorer has
        finished loading into VRAM. Auth-exempt; cheap to poll."""
        return {"status": "ok"}

    @app.get("/v1/models", tags=["introspection"], summary="List the loaded model")
    def models() -> dict[str, list[dict[str, object]]]:
        """The model id + revision currently loaded by the configured scorer.
        A single-element list — one Bernoulli server loads one model."""
        scorer_: Scorer = app.state.scorer
        return {
            "models": [
                {
                    "id": scorer_.model_id,
                    "revision": scorer_.revision,
                }
            ]
        }

    @app.post(
        "/v1/decide",
        response_model=DecideResponse,
        tags=["decide"],
        summary="Decide — score a lens pack over state",
    )
    def decide_endpoint(request: DecideRequest) -> DecideResponse:
        """The primary endpoint. One forward pass over the state produces a
        calibrated probability distribution per question, letting you branch on
        the probability (e.g. `if confidence > 0.9: auto_approve`) with the
        confidence number meaning what it says.

        Returns `DecideResponse` with one `Decision` per question (keyed by
        `question.id`), plus the model id, calibration version, and server-side
        latency in milliseconds."""
        try:
            return decide(
                request,
                cast(Scorer, app.state.scorer),
                calibration=app.state.calibration,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post(
        "/v1/generate",
        response_model=DecideResponse,
        tags=["decide"],
        summary="Generative baseline (comparison path)",
    )
    def generate_endpoint(request: DecideRequest) -> DecideResponse:
        """Same request shape as `/v1/decide`, but internally produces its
        distribution by generating a short completion and parsing it — the
        traditional "text-and-parse" baseline our benchmarks compare against.
        Returns a 1.0/0.0 point mass on the parsed answer (uniform on parse
        failure), not calibrated probabilities. Supported only when the
        loaded scorer has a `.generate()` method; a non-generating scorer
        gets 501."""
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
