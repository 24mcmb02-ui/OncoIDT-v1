"""
OncoIDT Hybrid Reasoner Service — FastAPI application.

Consumes onco:inference:score_update, applies clinical rule fusion,
and publishes annotated scores back to onco:inference:score_update
and onco:ward:state_change.

Endpoints:
  GET /reasoner/rules          — list active soft rules
  PUT /reasoner/rules/{id}     — update a soft rule (admin only)
  GET /health                  — liveness probe
  GET /ready                   — readiness probe

Requirements: 8.3, 8.6
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel

from shared.auth import Role, require_role
from shared.config import get_settings
from shared.db import close_engine, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, consume_events, get_redis, publish_event
from services.reasoner.fusion import FusionMode, FusionResult, fuse_scores, log_fusion_decision
from services.reasoner.rules import get_rule_engine, watch_rules_file

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

STREAM_SCORE = "onco:inference:score_update"
STREAM_WARD = "onco:ward:state_change"
CONSUMER_GROUP = "reasoner-service"
CONSUMER_NAME = "reasoner-0"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RuleUpdateRequest(BaseModel):
    enabled: bool | None = None
    score_delta: float | None = None
    condition_threshold: float | None = None
    description: str | None = None


class RulesResponse(BaseModel):
    rules: list[dict]
    count: int


# ---------------------------------------------------------------------------
# Feature snapshot fetch (best-effort)
# ---------------------------------------------------------------------------

async def _fetch_feature_snapshot(
    http_client: httpx.AsyncClient,
    patient_id: str,
    as_of: str | None,
) -> dict[str, Any]:
    """Fetch feature snapshot from the feature store; return {} on any failure."""
    try:
        url = f"{settings.feature_store_service_url}/features/{patient_id}"
        params = {"as_of": as_of} if as_of else {}
        resp = await http_client.get(url, params=params, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            # Feature store returns {"patient_id": ..., "features": {...}, ...}
            return data.get("features", data)
    except Exception as exc:
        logger.debug("Feature snapshot fetch failed for %s: %s", patient_id, exc)
    return {}


# ---------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------

async def run_reasoner_consumer() -> None:
    """
    Consume onco:inference:score_update, apply fusion, publish results.
    Runs indefinitely; exceptions are caught per-message to avoid crashing.
    """
    engine = get_rule_engine()
    session_factory = get_session_factory()

    async with httpx.AsyncClient() as http_client:
        async for msg_id, payload in consume_events(
            STREAM_SCORE,
            CONSUMER_GROUP,
            CONSUMER_NAME,
        ):
            try:
                # Skip survival estimates — no fusion needed
                if payload.get("event_type") == "survival_estimate":
                    continue

                # Skip already-annotated messages to avoid re-processing loops
                if payload.get("reasoner_annotated"):
                    continue

                patient_id: str = payload["patient_id"]
                score_type: str = payload["score_type"]
                timestamp: str = payload.get("timestamp", datetime.now(timezone.utc).isoformat())

                # Fetch feature snapshot (best-effort)
                features = await _fetch_feature_snapshot(
                    http_client,
                    patient_id,
                    as_of=timestamp,
                )

                # Evaluate clinical rules
                rule_overrides = engine.evaluate_rules(features)

                # Fuse ML score with rule overrides
                result: FusionResult = fuse_scores(payload, rule_overrides, FusionMode.HYBRID)

                # Log fusion decision to audit log
                async with session_factory() as session:
                    await log_fusion_decision(session, patient_id, result)
                    await session.commit()

                # Build annotated score payload
                annotated = {
                    **payload,
                    "ml_score": result.ml_score,
                    "final_score": result.final_score,
                    "rule_overrides": [
                        {
                            "rule_id": r.rule_id,
                            "threshold_value": r.threshold_value,
                            "triggered_value": r.triggered_value,
                            "score_floor": r.score_floor,
                        }
                        for r in result.rule_overrides
                    ],
                    "fusion_mode": result.fusion_mode,
                    "score": result.final_score,
                    "reasoner_annotated": True,
                }

                # Publish annotated score back to inference stream
                await publish_event(STREAM_SCORE, annotated)

                # Publish ward state change
                ward_event = {
                    "event_type": "score_update",
                    "patient_id": patient_id,
                    "score_type": score_type,
                    "forecast_horizon_hours": result.forecast_horizon_hours,
                    "final_score": result.final_score,
                    "fusion_mode": result.fusion_mode,
                    "timestamp": timestamp,
                }
                await publish_event(STREAM_WARD, ward_event)

                logger.debug(
                    "Fused score for patient=%s type=%s ml=%.3f final=%.3f mode=%s",
                    patient_id,
                    score_type,
                    result.ml_score,
                    result.final_score,
                    result.fusion_mode,
                )

            except Exception:
                logger.exception("Error processing score update message %s — skipping", msg_id)


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------

async def _check_redis() -> bool:
    try:
        redis = get_redis()
        await redis.ping()
        return True
    except Exception:
        return False


async def _check_rule_engine() -> bool:
    try:
        engine = get_rule_engine()
        engine.get_active_rules()  # len >= 0 always true if engine initialised
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    engine = get_rule_engine()

    import os
    rules_yaml_path = os.path.join(os.path.dirname(__file__), "rules.yaml")

    watcher_task = asyncio.create_task(
        watch_rules_file(rules_yaml_path, engine),
        name="reasoner-rules-watcher",
    )
    consumer_task = asyncio.create_task(
        run_reasoner_consumer(),
        name="reasoner-consumer",
    )

    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)

    logger.info("Reasoner service started.")
    yield

    await graceful_shutdown([consumer_task, watcher_task, error_monitor_task])

    await close_engine()
    await close_redis_pool()
    logger.info("Reasoner service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Reasoner Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="reasoner-service",
        readiness_checks={
            "redis": _check_redis,
            "rule_engine": _check_rule_engine,
        },
    )
)

# ---------------------------------------------------------------------------
# Rules router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/reasoner", tags=["reasoner"])


@router.get("/rules", response_model=RulesResponse, summary="List active clinical rules")
async def list_rules() -> RulesResponse:
    """Return all active soft rules from the rule engine."""
    engine = get_rule_engine()
    rules = engine.get_active_rules()
    return RulesResponse(rules=rules, count=len(rules))


@router.put(
    "/rules/{rule_id}",
    response_model=RulesResponse,
    summary="Update a soft rule (admin only)",
)
async def update_rule(
    rule_id: str,
    body: RuleUpdateRequest,
    _user: Annotated[Any, Depends(require_role(Role.SYSTEM_ADMINISTRATOR))],
) -> RulesResponse:
    """
    Partially update a soft rule by ID.  Triggers hot-reload of the rule
    engine by persisting the change to rules.yaml.
    Requires SYSTEM_ADMINISTRATOR role.
    """
    engine = get_rule_engine()
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update.")
    try:
        engine.update_soft_rule(rule_id, updates)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    rules = engine.get_active_rules()
    return RulesResponse(rules=rules, count=len(rules))


app.include_router(router)
