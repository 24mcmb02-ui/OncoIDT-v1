"""
OncoIDT API Gateway — Patient endpoints.

GET /api/v1/patients                    — list patients (ward-filtered)
GET /api/v1/patients/{id}               — patient twin state
GET /api/v1/patients/{id}/scores        — risk score history
GET /api/v1/patients/{id}/timeline      — full clinical timeline
GET /api/v1/patients/{id}/explanations  — latest SHAP explanations

All read endpoints target < 500ms under normal load.
Requirements: 13.1, 13.7, 17.6
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from shared.auth import CurrentUser, Role
from shared.config import get_settings
from shared.db import DbSession

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/patients", tags=["patients"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _assert_ward_access(session: DbSession, patient_id: str, current_user: Any) -> str:
    """Return ward_id or raise 403/404."""
    result = await session.execute(
        text("SELECT ward_id FROM patient_twins WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    ward_id: str = row[0]
    if not current_user.can_access_ward(ward_id):
        raise HTTPException(status_code=403, detail="Access denied to this patient's ward")
    return ward_id


# ---------------------------------------------------------------------------
# GET /api/v1/patients
# ---------------------------------------------------------------------------

@router.get("", summary="List patients (ward-filtered)")
async def list_patients(
    session: DbSession,
    current_user: CurrentUser,
    ward_id: str | None = Query(default=None),
    status: str | None = Query(default="active"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return patients accessible to the authenticated user, filtered by ward."""
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    # Ward access filter
    if current_user.role not in (Role.SYSTEM_ADMINISTRATOR, Role.INFECTION_CONTROL_OFFICER):
        if ward_id:
            if not current_user.can_access_ward(ward_id):
                raise HTTPException(status_code=403, detail="Access denied to this ward")
            conditions.append("ward_id = :ward_id")
            params["ward_id"] = ward_id
        elif current_user.ward_ids:
            conditions.append("ward_id = ANY(:ward_ids)")
            params["ward_ids"] = current_user.ward_ids
    elif ward_id:
        conditions.append("ward_id = :ward_id")
        params["ward_id"] = ward_id

    if status:
        conditions.append("status = :status")
        params["status"] = status

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    result = await session.execute(
        text(
            f"""
            SELECT patient_id, ward_id, bed_id, status, age_years, sex,
                   primary_diagnosis_icd10, chemo_regimen, chemo_cycle_phase,
                   immunosuppression_score, last_updated
            FROM patient_twins
            {where}
            ORDER BY last_updated DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    )
    rows = result.mappings().fetchall()

    count_result = await session.execute(
        text(f"SELECT COUNT(*) FROM patient_twins {where}"),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total: int = count_result.scalar_one()

    return {"patients": [dict(r) for r in rows], "total": total}


# ---------------------------------------------------------------------------
# GET /api/v1/patients/{id}
# ---------------------------------------------------------------------------

@router.get("/{patient_id}", summary="Patient twin state")
async def get_patient(
    patient_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Return the full PatientTwin state for a single patient."""
    await _assert_ward_access(session, patient_id, current_user)

    result = await session.execute(
        text("SELECT * FROM patient_twins WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    row = result.mappings().fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return dict(row)


# ---------------------------------------------------------------------------
# GET /api/v1/patients/{id}/scores
# ---------------------------------------------------------------------------

@router.get("/{patient_id}/scores", summary="Risk score history")
async def get_patient_scores(
    patient_id: str,
    session: DbSession,
    current_user: CurrentUser,
    score_type: str | None = Query(default=None, description="infection | deterioration"),
    horizon_hours: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Return risk score history for a patient."""
    await _assert_ward_access(session, patient_id, current_user)

    conditions = ["patient_id = :patient_id"]
    params: dict[str, Any] = {"patient_id": patient_id, "limit": limit}

    if score_type:
        conditions.append("score_type = :score_type")
        params["score_type"] = score_type
    if horizon_hours is not None:
        conditions.append("forecast_horizon_hours = :horizon_hours")
        params["horizon_hours"] = horizon_hours

    where = "WHERE " + " AND ".join(conditions)

    result = await session.execute(
        text(
            f"""
            SELECT score_id, score_type, forecast_horizon_hours, score,
                   uncertainty_lower, uncertainty_upper, model_version,
                   rule_overrides, timestamp
            FROM risk_scores
            {where}
            ORDER BY timestamp DESC
            LIMIT :limit
            """
        ),
        params,
    )
    rows = result.mappings().fetchall()
    return {"scores": [dict(r) for r in rows], "patient_id": patient_id}


# ---------------------------------------------------------------------------
# GET /api/v1/patients/{id}/timeline
# ---------------------------------------------------------------------------

@router.get("/{patient_id}/timeline", summary="Full clinical timeline")
async def get_patient_timeline(
    patient_id: str,
    session: DbSession,
    current_user: CurrentUser,
    record_type: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """Return the full clinical timeline (events, scores, alerts) for a patient."""
    await _assert_ward_access(session, patient_id, current_user)

    conditions = ["patient_id = :patient_id"]
    params: dict[str, Any] = {"patient_id": patient_id, "limit": limit}

    if record_type:
        conditions.append("record_type = :record_type")
        params["record_type"] = record_type

    where = "WHERE " + " AND ".join(conditions)

    events_result = await session.execute(
        text(
            f"""
            SELECT record_id, record_type, timestamp_utc, payload, data_quality_flags
            FROM clinical_events
            {where}
            ORDER BY timestamp_utc DESC
            LIMIT :limit
            """
        ),
        params,
    )
    events = [dict(r) for r in events_result.mappings().fetchall()]

    scores_result = await session.execute(
        text(
            """
            SELECT score_id, score_type, forecast_horizon_hours, score, timestamp
            FROM risk_scores
            WHERE patient_id = :patient_id
            ORDER BY timestamp DESC
            LIMIT 200
            """
        ),
        {"patient_id": patient_id},
    )
    scores = [dict(r) for r in scores_result.mappings().fetchall()]

    return {
        "patient_id": patient_id,
        "events": events,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/patients/{id}/explanations
# ---------------------------------------------------------------------------

@router.get("/{patient_id}/explanations", summary="Latest SHAP explanations")
async def get_patient_explanations(
    patient_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Return the latest SHAP feature attributions for a patient."""
    await _assert_ward_access(session, patient_id, current_user)

    # Proxy to explainability service
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.explainability_service_url}/explanations/{patient_id}",
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="No explanations found for this patient")
        raise HTTPException(status_code=502, detail="Explainability service error")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Explainability service timeout")
