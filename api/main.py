"""FastAPI service.

    POST /v1/score         one member
    POST /v1/score/batch   up to 1,000 members
    GET  /v1/model         model metadata, feature spec, test metrics
    GET  /healthz

Run from the repo root:

    .venv/bin/uvicorn api.main:app --port 8000

The model is loaded once, when the module is imported. Settings come from the
environment: RISKMODEL_MODEL_DIR and RISKMODEL_REPORTS_DIR (see riskmodel.config) and
RISKMODEL_AUDIT_DB (see api.audit).

This module does not use `from __future__ import annotations`: the request model is
built at startup from the feature spec, and FastAPI has to see the class itself in the
endpoint signatures, not its name as a string.
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import audit, schemas
from .service import ScoringService


def create_app(
    model_dir: Path | None = None,
    reports_dir: Path | None = None,
    audit_path: Path | None = None,
) -> FastAPI:
    service = ScoringService(model_dir, reports_dir)
    Features = schemas.build_feature_model(service.spec)

    class BatchRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        members: list[Features] = Field(min_length=1, max_length=schemas.MAX_BATCH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.audit = audit.AuditLog(audit_path)
        yield
        app.state.audit.close()

    app = FastAPI(
        title="Rising risk scoring API",
        version=service.model_version,
        description=(
            "Scores adult members for the chance that next year's cost of care lands in the "
            "top 10%. Trained on synthetic patients. Not for clinical or operational use."
        ),
        lifespan=lifespan,
    )
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Say where and why, but do not echo the rejected value. The default handler
        # does, which copies member data into responses and logs, and it cannot
        # serialize a NaN at all.
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def run(rows: list[list[float | None]]) -> tuple[str, list[dict]]:
        request_id = uuid.uuid4().hex
        results = service.score(rows)
        app.state.audit.append(
            request_id,
            service.model_version,
            [(r.pop("input_sha256"), r["probability"], r["tier"]) for r in results],
        )
        return request_id, results

    # Plain `def` endpoints run in FastAPI's thread pool, so scoring does not block the
    # event loop.
    @app.post("/v1/score", response_model=schemas.ScoreResponse)
    def score(features: Features):
        request_id, results = run([service.row(features.model_dump())])
        return {**results[0], "model_version": service.model_version, "request_id": request_id}

    @app.post("/v1/score/batch", response_model=schemas.BatchResponse)
    def score_batch(body: BatchRequest):
        request_id, results = run([service.row(m.model_dump()) for m in body.members])
        return {
            "results": [{"index": i, **r} for i, r in enumerate(results)],
            "n": len(results),
            "model_version": service.model_version,
            "request_id": request_id,
        }

    @app.get("/v1/model")
    def model():
        return service.model_info()

    @app.get("/healthz", response_model=schemas.Health)
    def healthz():
        return {
            "status": "ok",
            "model_version": service.model_version,
            "n_features": len(service.features),
        }

    return app


app = create_app()
