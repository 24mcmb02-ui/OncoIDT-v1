"""
OncoIDT API Gateway — Research export endpoint.

GET /api/v1/research/export — de-identified patient data export
  - Research_Analyst role only
  - Applies HIPAA Safe Harbor 18-identifier removal via deidentify.py
  - Returns de-identified PatientTwin list

Requirements: 13.1, 17.6
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text

from ..deidentify import deidentify_patient_list
from shared.auth import Role, TokenPayload, require_role
from shared.config import get_settings
from shared.db import DbSession
from shared.schemas import PatientTwinSchema

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/research", tags=["research"])


@router.get(
    "/export",
    summary="De-identified patient data export (Research_Analyst only)",
)
async def research_export(
    session: DbSession,
    current_user: TokenPayload = Depends(require_role(Role.RESEARCH_ANALYST)),
    ward_id: str | None = Query(default=None, description="Filter by ward (optional)"),
    status: str | None = Query(default="active"),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """
    Export de-identified patient data for research use.
    All 18 HIPAA Safe Harbor identifiers are removed before returning data.
    Only accessible to Research_Analyst role.
    """
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if ward_id:
        conditions.append("ward_id = :ward_id")
        params["ward_id"] = ward_id
    if status:
        conditions.append("status = :status")
        params["status"] = status

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    result = await session.execute(
        text(
            f"""
            SELECT *
            FROM patient_twins
            {where}
            ORDER BY admission_timestamp DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    )
    rows = result.mappings().fetchall()

    # Parse into Pydantic schemas for de-identification
    twins: list[PatientTwinSchema] = []
    for row in rows:
        try:
            twins.append(PatientTwinSchema.model_validate(dict(row)))
        except Exception as exc:
            logger.warning("Skipping malformed patient row: %s", exc)

    deidentified = deidentify_patient_list(twins)

    count_result = await session.execute(
        text(f"SELECT COUNT(*) FROM patient_twins {where}"),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total: int = count_result.scalar_one()

    return {
        "patients": deidentified,
        "total": total,
        "deidentified": True,
        "method": "hipaa_safe_harbor",
    }
