"""
OncoIDT Inference Service — FastAPI application.

Endpoints:
  POST /inference/score/{patient_id}
       On-demand single-patient scoring.

  GET  /inference/scores/{patient_id}
       Retrieve score history from the ``risk_scores`` table.

  GET  /health  — liveness probe
  GET  /ready   — readiness probe

Background tasks (started via FastAPI lifespan):
  - Redis Stream event consumer (scorer.run_event_consumer)
  - APScheduler batch re-scorer (batch_rescorer.create_batch_rescorer)
  - ModelSlot background polling thread (model_slot.ModelSlot.start_polling)

Requirements: 4.1, 4.2, 5.1
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from shared.config import get_settings
from shared.db import close_engine, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, get_redis
from services.inference.batch_rescorer import create_batch_rescorer
from services.inference.model_slot import get_model_slot
from services.inference.scorer import STREAM_OUT, score_patient, run_event_consumer

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class ScoreResponse(BaseModel):
    patient_id: str
    scores: list[dict[str, Any]]
    model_version: str | None = None


class ScoreHistoryItem(BaseModel):
    score_id: str
    score_type: str
    forecast_horizon_hours: int
    score: float
    uncertainty_lower: float
    uncertainty_upper: float
    model_version: str
    staleness_flag: bool = False
    timestamp: datetime


class ScoreHistoryResponse(BaseModel):
    patient_id: str
    count: int
    scores: list[ScoreHistoryItem]


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------

async def _check_db() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        redis = get_redis()
        await redis.ping()
        return True
    except Exception:
        return False


def _check_model() -> bool:
    return get_model_slot().current_model is not None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    slot = get_model_slot()

    # Load initial model (best-effort; service starts even without a model)
    slot.load_initial()

    # Start background model polling thread
    slot.start_polling()

    # Start batch re-scorer
    scheduler = create_batch_rescorer(slot)
    scheduler.start()

    # Start event consumer as an asyncio task
    consumer_task = asyncio.create_task(
        run_event_consumer(slot),
        name="inference-event-consumer",
    )

    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)

    logger.info("Inference service started.")
    yield

    # Shutdown
    await graceful_shutdown([consumer_task, error_monitor_task])

    scheduler.shutdown(wait=False)
    slot.stop_polling()
    await close_engine()
    await close_redis_pool()
    logger.info("Inference service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Inference Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="inference-service",
        readiness_checks={
            "db": _check_db,
            "redis": _check_redis,
        },
    )
)


# ---------------------------------------------------------------------------
# POST /inference/score/{patient_id}
# ---------------------------------------------------------------------------

@app.post(
    "/inference/score/{patient_id}",
    response_model=ScoreResponse,
    summary="On-demand single-patient scoring",
)
async def score_patient_endpoint(patient_id: str) -> ScoreResponse:
    """
    Trigger immediate inference for a single patient and return the
    resulting risk scores.  Also publishes scores to the Redis stream.
    Requirements: 4.1, 4.2, 5.1
    """
    slot = get_model_slot()
    if slot.current_model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Inference service is not ready.",
        )

    redis = get_redis()
    async with httpx.AsyncClient() as http_client:
        payloads = await score_patient(patient_id, slot, http_client)

    if not payloads:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed for patient {patient_id}",
        )

    # Publish to stream
    for payload in payloads:
        try:
            from shared.redis_client import publish_event
            await publish_event(STREAM_OUT, payload, redis=redis)
        except Exception as exc:
            logger.warning("Failed to publish score to stream: %s", exc)

    return ScoreResponse(
        patient_id=patient_id,
        scores=payloads,
        model_version=slot.current_version,
    )


# ---------------------------------------------------------------------------
# GET /inference/scores/{patient_id}
# ---------------------------------------------------------------------------

@app.get(
    "/inference/scores/{patient_id}",
    response_model=ScoreHistoryResponse,
    summary="Retrieve score history for a patient",
)
async def get_score_history(
    patient_id: str,
    score_type: Annotated[
        str | None,
        Query(description="Filter by score type: 'infection' or 'deterioration'"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of scores to return"),
    ] = 100,
) -> ScoreHistoryResponse:
    """
    Retrieve the risk score history for a patient from the ``risk_scores``
    table.  Requirements: 4.1, 4.2.
    """
    factory = get_session_factory()
    async with factory() as session:
        query = """
            SELECT score_id, score_type, forecast_horizon_hours,
                   score, uncertainty_lower, uncertainty_upper,
                   model_version, timestamp
            FROM risk_scores
            WHERE patient_id = :patient_id
        """
        params: dict[str, Any] = {"patient_id": patient_id}

        if score_type:
            query += " AND score_type = :score_type"
            params["score_type"] = score_type

        query += " ORDER BY timestamp DESC LIMIT :limit"
        params["limit"] = limit

        result = await session.execute(text(query), params)
        rows = result.fetchall()

    scores = [
        ScoreHistoryItem(
            score_id=str(row[0]),
            score_type=str(row[1]),
            forecast_horizon_hours=int(row[2]),
            score=float(row[3]),
            uncertainty_lower=float(row[4]) if row[4] is not None else 0.0,
            uncertainty_upper=float(row[5]) if row[5] is not None else 1.0,
            model_version=str(row[6]),
            timestamp=row[7],
        )
        for row in rows
    ]

    return ScoreHistoryResponse(
        patient_id=patient_id,
        count=len(scores),
        scores=scores,
    )
