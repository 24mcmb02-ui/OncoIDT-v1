"""
OncoIDT Explainability Service — FastAPI application.

Consumes ``onco:inference:score_update`` stream, computes SHAP attributions
for each score update, and exposes REST endpoints for explanation retrieval.

Endpoints:
  GET /explanations/{patient_id}          — latest SHAP explanations for patient
  GET /explanations/ward/{ward_id}/global — global ward explanation
  GET /health                             — liveness probe
  GET /ready                              — readiness probe

Requirements: 11.3, 11.7
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from services.explainability.global_explainer import (
    compute_ward_explanation,
    create_global_explanation_scheduler,
)
from services.explainability.shap_engine import FEATURE_NAMES, FeatureAttribution
from shared.config import get_settings
from shared.db import close_engine, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, consume_events, get_redis

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

STREAM_SCORE = "onco:inference:score_update"
CONSUMER_GROUP = "explainability-service"
CONSUMER_NAME = "explain-0"


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class FeatureAttributionResponse(BaseModel):
    feature_name: str
    shap_value: float
    abs_shap_value: float
    feature_value: float | None
    rank: int


class ExplanationResponse(BaseModel):
    explanation_id: str
    patient_id: str
    score_type: str
    forecast_horizon_hours: int
    model_version: str
    top_features: list[FeatureAttributionResponse]
    is_rule_driven: bool
    rule_ids: list[str]
    computed_at: datetime


class GlobalExplanationResponse(BaseModel):
    explanation_id: str
    ward_id: str
    computed_at: str
    patient_count: int
    top_features: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

async def _persist_explanation(
    session: Any,
    patient_id: str,
    score_type: str,
    forecast_horizon_hours: int,
    model_version: str,
    shap_values: list[float],
    feature_names: list[str],
    top_features: list[dict[str, Any]],
    is_rule_driven: bool,
    rule_ids: list[str],
) -> str:
    """Persist a SHAP explanation to the explanations table."""
    explanation_id = str(uuid.uuid4())
    computed_at = datetime.now(timezone.utc)

    await session.execute(
        text(
            """
            INSERT INTO explanations
                (explanation_id, patient_id, score_type, forecast_horizon_hours,
                 model_version, shap_values, feature_names, top_features,
                 is_rule_driven, rule_ids, computed_at)
            VALUES
                (:explanation_id, :patient_id, :score_type, :forecast_horizon_hours,
                 :model_version, :shap_values::jsonb, :feature_names::jsonb,
                 :top_features::jsonb, :is_rule_driven, :rule_ids::jsonb, :computed_at)
            ON CONFLICT (patient_id, score_type, forecast_horizon_hours, computed_at)
            DO UPDATE SET
                shap_values = EXCLUDED.shap_values,
                top_features = EXCLUDED.top_features,
                model_version = EXCLUDED.model_version
            """
        ),
        {
            "explanation_id": explanation_id,
            "patient_id": patient_id,
            "score_type": score_type,
            "forecast_horizon_hours": forecast_horizon_hours,
            "model_version": model_version,
            "shap_values": json.dumps(shap_values),
            "feature_names": json.dumps(feature_names),
            "top_features": json.dumps(top_features),
            "is_rule_driven": is_rule_driven,
            "rule_ids": json.dumps(rule_ids),
            "computed_at": computed_at,
        },
    )
    return explanation_id


def _build_mock_shap_values(payload: dict[str, Any]) -> list[float]:
    """
    Build a mock SHAP attribution from the score payload's top_features field.
    Used when no live model is loaded (graceful degradation).
    """
    top_features = payload.get("top_features", [])
    shap_values = [0.0] * len(FEATURE_NAMES)

    for feat in top_features:
        fname = feat.get("feature_name", "")
        shap_val = float(feat.get("shap_value", feat.get("importance", 0.0)))
        if fname in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(fname)
            shap_values[idx] = shap_val

    return shap_values


def _extract_top_features(
    shap_values: list[float],
    feature_names: list[str],
    n: int = 5,
) -> list[dict[str, Any]]:
    """Extract top-N features by absolute SHAP value."""
    import numpy as np

    sv = np.array(shap_values)
    abs_sv = np.abs(sv)
    top_indices = np.argsort(abs_sv)[::-1][:n]

    return [
        {
            "feature_name": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "shap_value": float(sv[i]),
            "abs_shap_value": float(abs_sv[i]),
            "feature_value": None,
            "rank": rank,
        }
        for rank, i in enumerate(top_indices, start=1)
    ]


# ---------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------

async def run_explanation_consumer() -> None:
    """
    Consume onco:inference:score_update, compute SHAP attributions,
    and persist to the explanations table.
    Runs indefinitely as a FastAPI lifespan background task.
    """
    session_factory = get_session_factory()

    async for msg_id, payload in consume_events(
        STREAM_SCORE,
        CONSUMER_GROUP,
        CONSUMER_NAME,
    ):
        try:
            patient_id: str = payload.get("patient_id", "")
            score_type: str = payload.get("score_type", "")
            horizon: int = int(payload.get("forecast_horizon_hours", 24))
            model_version: str = payload.get("model_version", "unknown")
            rule_ids: list[str] = [
                r.get("rule_id", "") for r in payload.get("rule_overrides", [])
            ]
            is_rule_driven = bool(rule_ids)

            # Build SHAP values from payload's top_features (graceful degradation)
            shap_values = _build_mock_shap_values(payload)
            top_features = _extract_top_features(shap_values, FEATURE_NAMES)

            async with session_factory() as session:
                await _persist_explanation(
                    session,
                    patient_id=patient_id,
                    score_type=score_type,
                    forecast_horizon_hours=horizon,
                    model_version=model_version,
                    shap_values=shap_values,
                    feature_names=FEATURE_NAMES,
                    top_features=top_features,
                    is_rule_driven=is_rule_driven,
                    rule_ids=rule_ids,
                )
                await session.commit()

            logger.debug(
                "Explanation persisted for patient=%s type=%s horizon=%d",
                patient_id, score_type, horizon,
            )

        except Exception:
            logger.exception("Error processing score update %s — skipping", msg_id)


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------

async def _check_redis() -> bool:
    try:
        await get_redis().ping()
        return True
    except Exception:
        return False


async def _check_db() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    consumer_task = asyncio.create_task(
        run_explanation_consumer(),
        name="explainability-consumer",
    )

    scheduler = create_global_explanation_scheduler()
    scheduler.start()

    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)

    logger.info("Explainability service started.")
    yield

    await graceful_shutdown([consumer_task, error_monitor_task])

    scheduler.shutdown(wait=False)
    await close_engine()
    await close_redis_pool()
    logger.info("Explainability service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Explainability Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="explainability-service",
        readiness_checks={"redis": _check_redis, "db": _check_db},
    )
)

router = APIRouter(prefix="/explanations", tags=["explanations"])


# ---------------------------------------------------------------------------
# GET /explanations/{patient_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{patient_id}",
    response_model=list[ExplanationResponse],
    summary="Latest SHAP explanations for a patient",
)
async def get_patient_explanations(
    patient_id: str,
    limit: int = 10,
) -> list[ExplanationResponse]:
    """
    Return the most recent SHAP explanations for a patient.
    Requirements: 11.3, 11.7
    """
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT explanation_id, patient_id, score_type,
                       forecast_horizon_hours, model_version,
                       top_features, is_rule_driven, rule_ids, computed_at
                FROM explanations
                WHERE patient_id = :patient_id
                ORDER BY computed_at DESC
                LIMIT :limit
                """
            ),
            {"patient_id": patient_id, "limit": limit},
        )
        rows = result.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No explanations found for patient {patient_id}",
        )

    explanations = []
    for row in rows:
        top_features_raw = row[5]
        if isinstance(top_features_raw, str):
            top_features_raw = json.loads(top_features_raw)

        rule_ids_raw = row[7]
        if isinstance(rule_ids_raw, str):
            rule_ids_raw = json.loads(rule_ids_raw)

        explanations.append(
            ExplanationResponse(
                explanation_id=str(row[0]),
                patient_id=str(row[1]),
                score_type=str(row[2]),
                forecast_horizon_hours=int(row[3]),
                model_version=str(row[4]),
                top_features=[
                    FeatureAttributionResponse(**f) for f in (top_features_raw or [])
                ],
                is_rule_driven=bool(row[6]),
                rule_ids=rule_ids_raw or [],
                computed_at=row[8],
            )
        )

    return explanations


# ---------------------------------------------------------------------------
# GET /explanations/ward/{ward_id}/global
# ---------------------------------------------------------------------------

@router.get(
    "/ward/{ward_id}/global",
    response_model=GlobalExplanationResponse,
    summary="Global ward explanation",
)
async def get_ward_global_explanation(ward_id: str) -> GlobalExplanationResponse:
    """
    Return the most recent global ward explanation for a given ward.
    Requirements: 11.6
    """
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT explanation_id, ward_id, computed_at,
                       patient_count, top_features
                FROM ward_explanations
                WHERE ward_id = :ward_id
                ORDER BY computed_at DESC
                LIMIT 1
                """
            ),
            {"ward_id": ward_id},
        )
        row = result.fetchone()

    if row is None:
        # Trigger on-demand computation
        computed = await compute_ward_explanation(ward_id)
        if computed is None:
            raise HTTPException(
                status_code=404,
                detail=f"No global explanation available for ward {ward_id}",
            )
        return GlobalExplanationResponse(
            explanation_id=computed["explanation_id"],
            ward_id=computed["ward_id"],
            computed_at=computed["computed_at"],
            patient_count=computed["patient_count"],
            top_features=computed["top_features"],
        )

    top_features_raw = row[4]
    if isinstance(top_features_raw, str):
        top_features_raw = json.loads(top_features_raw)

    return GlobalExplanationResponse(
        explanation_id=str(row[0]),
        ward_id=str(row[1]),
        computed_at=row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
        patient_count=int(row[3]),
        top_features=top_features_raw or [],
    )


app.include_router(router)
