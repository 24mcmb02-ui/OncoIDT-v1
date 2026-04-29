"""
OncoIDT Alert Service — FastAPI Application.

Consumes ``onco:inference:score_update`` stream, runs the full
generator → dedup → router pipeline, and exposes REST endpoints for
alert management.

Endpoints:
  GET  /alerts                      — role-filtered alert feed
  POST /alerts/{id}/acknowledge     — acknowledge an alert
  POST /alerts/{id}/snooze          — snooze an alert (mandatory reason)
  POST /alerts/{id}/escalate        — escalate an alert to Critical
  GET  /health                      — liveness probe
  GET  /ready                       — readiness probe

Alert performance log (Requirement 10.7):
  Writes alert-to-action latency and volume per shift to
  ``alert_performance_log`` table.

Requirements: 10.7
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from services.alerts.actions import (
    acknowledge,
    cancel_all_snooze_tasks,
    escalate,
    snooze,
)
from services.alerts.dedup import deduplicate_alert
from services.alerts.generator import (
    Alert,
    evaluate_score_update,
    generate_source_unavailability_alert,
    generate_ward_exposure_alert,
)
from services.alerts.router import route_alert
from shared.auth import CurrentUser, Role, require_role
from shared.config import get_settings
from shared.db import DbSession, close_engine, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import (
    close_redis_pool,
    consume_events,
    get_redis,
    publish_event,
)

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

STREAM_SCORE = "onco:inference:score_update"
CONSUMER_GROUP = "alert-service"
CONSUMER_NAME = "alert-0"

# In-process cache of last known scores per (patient_id, score_type, horizon)
# Used for delta rate computation
_score_cache: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class SnoozeRequest(BaseModel):
    reason: str = Field(..., min_length=1, description="Mandatory reason for snooze")
    duration_minutes: int = Field(default=30, ge=1, le=480)


class AlertResponse(BaseModel):
    alert_id: str
    patient_id: str | None
    ward_id: str | None
    alert_type: str
    priority: str
    score: float | None
    score_delta: float | None
    message: str
    escalation_count: int
    acknowledged: bool
    snoozed_until: datetime | None
    generated_at: datetime
    top_features: list[dict[str, Any]]


class ActionResponse(BaseModel):
    action_id: str
    alert_id: str
    status: str


class AlertFeedResponse(BaseModel):
    alerts: list[AlertResponse]
    total: int


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _persist_alert(session: Any, alert: Alert) -> None:
    """Insert or update an alert row in the alerts table."""
    await session.execute(
        text(
            """
            INSERT INTO alerts (
                alert_id, patient_id, ward_id, alert_type, priority,
                score, score_delta, message, escalation_count,
                acknowledged, snoozed_until, generated_at, top_features
            ) VALUES (
                :alert_id, :patient_id, :ward_id, :alert_type, :priority,
                :score, :score_delta, :message, :escalation_count,
                :acknowledged, :snoozed_until, :generated_at, :top_features::jsonb
            )
            ON CONFLICT (alert_id) DO UPDATE SET
                escalation_count = EXCLUDED.escalation_count,
                generated_at = EXCLUDED.generated_at,
                score = EXCLUDED.score
            """
        ),
        {
            "alert_id": alert.alert_id,
            "patient_id": alert.patient_id,
            "ward_id": alert.ward_id,
            "alert_type": alert.alert_type,
            "priority": alert.priority,
            "score": alert.score,
            "score_delta": alert.score_delta,
            "message": alert.message,
            "escalation_count": alert.escalation_count,
            "acknowledged": alert.acknowledged,
            "snoozed_until": alert.snoozed_until,
            "generated_at": alert.generated_at,
            "top_features": __import__("json").dumps(alert.top_features),
        },
    )


async def _log_alert_performance(
    session: Any,
    alert_id: str,
    action_type: str,
    action_at: datetime,
    generated_at: datetime,
    ward_id: str | None,
) -> None:
    """Write alert-to-action latency to alert_performance_log."""
    latency_seconds = (action_at - generated_at).total_seconds()
    shift_label = _current_shift_label(action_at)
    await session.execute(
        text(
            """
            INSERT INTO alert_performance_log
                (log_id, alert_id, action_type, generated_at, action_at,
                 latency_seconds, shift_label, ward_id)
            VALUES
                (:log_id, :alert_id, :action_type, :generated_at, :action_at,
                 :latency_seconds, :shift_label, :ward_id)
            """
        ),
        {
            "log_id": str(uuid.uuid4()),
            "alert_id": alert_id,
            "action_type": action_type,
            "generated_at": generated_at,
            "action_at": action_at,
            "latency_seconds": latency_seconds,
            "shift_label": shift_label,
            "ward_id": ward_id,
        },
    )


def _current_shift_label(dt: datetime) -> str:
    """Return a shift label (day/evening/night) for a given datetime."""
    hour = dt.hour
    if 7 <= hour < 15:
        return "day"
    elif 15 <= hour < 23:
        return "evening"
    else:
        return "night"


async def _fetch_features(
    http_client: httpx.AsyncClient,
    patient_id: str,
    as_of: str | None,
) -> dict[str, Any]:
    """Fetch feature snapshot from feature store; return {} on failure."""
    try:
        url = f"{settings.feature_store_service_url}/features/{patient_id}"
        params = {"as_of": as_of} if as_of else {}
        resp = await http_client.get(url, params=params, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("features", data)
    except Exception as exc:
        logger.debug("Feature fetch failed for %s: %s", patient_id, exc)
    return {}


# ---------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------

async def run_alert_consumer() -> None:
    """
    Consume onco:inference:score_update, run generator → dedup → router.
    Runs indefinitely as a FastAPI lifespan background task.
    """
    redis = get_redis()
    session_factory = get_session_factory()

    async with httpx.AsyncClient() as http_client:
        async for msg_id, payload in consume_events(
            STREAM_SCORE,
            CONSUMER_GROUP,
            CONSUMER_NAME,
            redis=redis,
        ):
            try:
                # Skip survival estimates — no alert thresholds apply
                if payload.get("event_type") == "survival_estimate":
                    continue

                patient_id: str = payload.get("patient_id", "")
                score_type: str = payload.get("score_type", "")
                horizon: int = int(payload.get("forecast_horizon_hours", 24))
                timestamp: str = payload.get("timestamp", datetime.now(timezone.utc).isoformat())

                # Retrieve previous score for delta computation
                cache_key = f"{patient_id}:{score_type}:{horizon}"
                previous_score = _score_cache.get(cache_key)
                current_score = float(payload.get("final_score", payload.get("score", 0.0)))
                _score_cache[cache_key] = current_score

                # Fetch feature snapshot (best-effort)
                features = await _fetch_features(http_client, patient_id, as_of=timestamp)

                # Generate alerts
                alerts = evaluate_score_update(payload, features, previous_score)

                for alert in alerts:
                    # Dedup
                    alert, is_dup = await deduplicate_alert(alert, redis)

                    # Persist
                    async with session_factory() as session:
                        await _persist_alert(session, alert)
                        await session.commit()

                    # Route → publish to onco:alert:generated
                    await route_alert(alert, redis, is_duplicate=is_dup)

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
        run_alert_consumer(),
        name="alert-consumer",
    )
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    logger.info("Alert service started.")
    yield

    await graceful_shutdown([consumer_task, error_monitor_task])

    cancel_all_snooze_tasks()
    await close_engine()
    await close_redis_pool()
    logger.info("Alert service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Alert Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="alert-service",
        readiness_checks={"redis": _check_redis, "db": _check_db},
    )
)

router = APIRouter(prefix="/alerts", tags=["alerts"])


# ---------------------------------------------------------------------------
# GET /alerts — role-filtered alert feed
# ---------------------------------------------------------------------------

@router.get("", response_model=AlertFeedResponse, summary="Role-filtered alert feed")
async def list_alerts(
    session: DbSession,
    current_user: CurrentUser,
    ward_id: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    alert_type: str | None = Query(default=None),
    acknowledged: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AlertFeedResponse:
    """
    Return alerts filtered by the authenticated user's role and ward access.
    Clinicians and Charge Nurses see only their ward(s).
    Infection Control Officers and Admins see all wards.
    """
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    # Ward access filter
    if not current_user.role in (Role.SYSTEM_ADMINISTRATOR, Role.INFECTION_CONTROL_OFFICER):
        if current_user.ward_ids:
            conditions.append("(ward_id = ANY(:ward_ids) OR ward_id IS NULL)")
            params["ward_ids"] = current_user.ward_ids
        elif ward_id:
            conditions.append("ward_id = :ward_id")
            params["ward_id"] = ward_id

    if ward_id and current_user.role in (Role.SYSTEM_ADMINISTRATOR, Role.INFECTION_CONTROL_OFFICER):
        conditions.append("ward_id = :ward_id")
        params["ward_id"] = ward_id

    if priority:
        conditions.append("priority = :priority")
        params["priority"] = priority

    if alert_type:
        conditions.append("alert_type = :alert_type")
        params["alert_type"] = alert_type

    if acknowledged is not None:
        conditions.append("acknowledged = :acknowledged")
        params["acknowledged"] = acknowledged

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    result = await session.execute(
        text(
            f"""
            SELECT alert_id, patient_id, ward_id, alert_type, priority,
                   score, score_delta, message, escalation_count,
                   acknowledged, snoozed_until, generated_at, top_features
            FROM alerts
            {where_clause}
            ORDER BY generated_at DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    )
    rows = result.mappings().fetchall()

    count_result = await session.execute(
        text(f"SELECT COUNT(*) FROM alerts {where_clause}"),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total: int = count_result.scalar_one()

    import json as _json

    alerts_out = [
        AlertResponse(
            alert_id=row["alert_id"],
            patient_id=row["patient_id"],
            ward_id=row["ward_id"],
            alert_type=row["alert_type"],
            priority=row["priority"],
            score=row["score"],
            score_delta=row["score_delta"],
            message=row["message"],
            escalation_count=row["escalation_count"],
            acknowledged=row["acknowledged"],
            snoozed_until=row["snoozed_until"],
            generated_at=row["generated_at"],
            top_features=_json.loads(row["top_features"]) if row["top_features"] else [],
        )
        for row in rows
    ]
    return AlertFeedResponse(alerts=alerts_out, total=total)


# ---------------------------------------------------------------------------
# POST /alerts/{id}/acknowledge
# ---------------------------------------------------------------------------

@router.post("/{alert_id}/acknowledge", response_model=ActionResponse)
async def acknowledge_alert(
    alert_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> ActionResponse:
    """Acknowledge an alert. Records action and latency in performance log."""
    alert_row = await _get_alert_or_404(session, alert_id)
    action_id = await acknowledge(alert_id, current_user.user_id, session, get_redis())

    await _log_alert_performance(
        session,
        alert_id=alert_id,
        action_type="acknowledge",
        action_at=datetime.now(timezone.utc),
        generated_at=alert_row["generated_at"],
        ward_id=alert_row.get("ward_id"),
    )
    await session.commit()
    return ActionResponse(action_id=action_id, alert_id=alert_id, status="acknowledged")


# ---------------------------------------------------------------------------
# POST /alerts/{id}/snooze
# ---------------------------------------------------------------------------

@router.post("/{alert_id}/snooze", response_model=ActionResponse)
async def snooze_alert(
    alert_id: str,
    body: SnoozeRequest,
    session: DbSession,
    current_user: CurrentUser,
) -> ActionResponse:
    """Snooze an alert with a mandatory reason. Schedules re-evaluation at expiry."""
    alert_row = await _get_alert_or_404(session, alert_id)
    action_id = await snooze(
        alert_id,
        current_user.user_id,
        body.reason,
        body.duration_minutes,
        session,
        get_redis(),
    )

    await _log_alert_performance(
        session,
        alert_id=alert_id,
        action_type="snooze",
        action_at=datetime.now(timezone.utc),
        generated_at=alert_row["generated_at"],
        ward_id=alert_row.get("ward_id"),
    )
    await session.commit()
    return ActionResponse(action_id=action_id, alert_id=alert_id, status="snoozed")


# ---------------------------------------------------------------------------
# POST /alerts/{id}/escalate
# ---------------------------------------------------------------------------

@router.post(
    "/{alert_id}/escalate",
    response_model=ActionResponse,
)
async def escalate_alert(
    alert_id: str,
    session: DbSession,
    current_user: Annotated[Any, Depends(require_role(
        Role.CLINICIAN, Role.CHARGE_NURSE, Role.INFECTION_CONTROL_OFFICER, Role.SYSTEM_ADMINISTRATOR
    ))],
) -> ActionResponse:
    """Escalate an alert to Critical priority."""
    alert_row = await _get_alert_or_404(session, alert_id)
    action_id = await escalate(alert_id, current_user.user_id, session, get_redis())

    await _log_alert_performance(
        session,
        alert_id=alert_id,
        action_type="escalate",
        action_at=datetime.now(timezone.utc),
        generated_at=alert_row["generated_at"],
        ward_id=alert_row.get("ward_id"),
    )
    await session.commit()
    return ActionResponse(action_id=action_id, alert_id=alert_id, status="escalated")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

async def _get_alert_or_404(session: Any, alert_id: str) -> Any:
    result = await session.execute(
        text("SELECT * FROM alerts WHERE alert_id = :alert_id"),
        {"alert_id": alert_id},
    )
    row = result.mappings().fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return row


app.include_router(router)
