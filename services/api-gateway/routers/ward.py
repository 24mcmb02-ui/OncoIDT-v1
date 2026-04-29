"""
OncoIDT API Gateway — Ward endpoints.

GET /api/v1/ward/{id}         — ward twin state + heatmap
GET /api/v1/ward/{id}/alerts  — active alerts for ward

Requirements: 13.1, 13.7
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from shared.auth import CurrentUser
from shared.config import get_settings
from shared.db import DbSession

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/ward", tags=["ward"])


@router.get("/{ward_id}", summary="Ward twin state and heatmap")
async def get_ward(
    ward_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Return the WardTwin state including per-bed risk heatmap."""
    if not current_user.can_access_ward(ward_id):
        raise HTTPException(status_code=403, detail="Access denied to this ward")

    # Aggregate ward stats from patient_twins
    result = await session.execute(
        text(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'active') AS occupied_beds,
                AVG(
                    CASE WHEN status = 'active'
                    THEN (SELECT MAX(score) FROM risk_scores rs
                          WHERE rs.patient_id = pt.patient_id
                            AND rs.score_type = 'infection'
                          LIMIT 1)
                    END
                ) AS ward_infection_risk,
                AVG(
                    CASE WHEN status = 'active'
                    THEN (SELECT MAX(score) FROM risk_scores rs
                          WHERE rs.patient_id = pt.patient_id
                            AND rs.score_type = 'deterioration'
                          LIMIT 1)
                    END
                ) AS ward_deterioration_risk,
                COUNT(*) FILTER (
                    WHERE status = 'active'
                    AND EXISTS (
                        SELECT 1 FROM risk_scores rs
                        WHERE rs.patient_id = pt.patient_id
                          AND rs.score_type = 'infection'
                          AND rs.score > 0.6
                    )
                ) AS high_risk_patient_count
            FROM patient_twins pt
            WHERE ward_id = :ward_id
            """
        ),
        {"ward_id": ward_id},
    )
    stats = result.mappings().fetchone() or {}

    # Per-bed state
    beds_result = await session.execute(
        text(
            """
            SELECT pt.patient_id, pt.bed_id,
                   (SELECT score FROM risk_scores rs
                    WHERE rs.patient_id = pt.patient_id AND rs.score_type = 'infection'
                    ORDER BY rs.timestamp DESC LIMIT 1) AS infection_risk_score,
                   (SELECT score FROM risk_scores rs
                    WHERE rs.patient_id = pt.patient_id AND rs.score_type = 'deterioration'
                    ORDER BY rs.timestamp DESC LIMIT 1) AS deterioration_risk_score,
                   pt.last_updated
            FROM patient_twins pt
            WHERE pt.ward_id = :ward_id AND pt.status = 'active'
            """
        ),
        {"ward_id": ward_id},
    )
    beds = [dict(r) for r in beds_result.mappings().fetchall()]

    return {
        "ward_id": ward_id,
        "occupied_beds": stats.get("occupied_beds", 0),
        "ward_infection_risk": stats.get("ward_infection_risk") or 0.0,
        "ward_deterioration_risk": stats.get("ward_deterioration_risk") or 0.0,
        "high_risk_patient_count": stats.get("high_risk_patient_count", 0),
        "beds": beds,
    }


@router.get("/{ward_id}/alerts", summary="Active alerts for ward")
async def get_ward_alerts(
    ward_id: str,
    session: DbSession,
    current_user: CurrentUser,
    priority: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return active (unacknowledged) alerts for a ward."""
    if not current_user.can_access_ward(ward_id):
        raise HTTPException(status_code=403, detail="Access denied to this ward")

    conditions = ["ward_id = :ward_id", "acknowledged = false"]
    params: dict[str, Any] = {"ward_id": ward_id, "limit": limit}

    if priority:
        conditions.append("priority = :priority")
        params["priority"] = priority

    where = "WHERE " + " AND ".join(conditions)

    result = await session.execute(
        text(
            f"""
            SELECT alert_id, patient_id, alert_type, priority, score,
                   message, escalation_count, generated_at
            FROM alerts
            {where}
            ORDER BY
                CASE priority
                    WHEN 'Critical' THEN 1
                    WHEN 'High' THEN 2
                    WHEN 'Medium' THEN 3
                    ELSE 4
                END,
                generated_at DESC
            LIMIT :limit
            """
        ),
        params,
    )
    rows = result.mappings().fetchall()
    return {"ward_id": ward_id, "alerts": [dict(r) for r in rows]}
