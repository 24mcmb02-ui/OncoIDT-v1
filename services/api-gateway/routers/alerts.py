"""
OncoIDT API Gateway — Alert management endpoints.

GET  /api/v1/alerts                      — role-filtered alert feed
POST /api/v1/alerts/{id}/acknowledge     — acknowledge alert
POST /api/v1/alerts/{id}/snooze          — snooze alert (mandatory reason)
POST /api/v1/alerts/{id}/escalate        — escalate alert

Proxies to alert-service.
Requirements: 13.1, 13.7
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from shared.auth import CurrentUser
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class SnoozeRequest(BaseModel):
    reason: str = Field(..., min_length=1)
    duration_minutes: int = Field(default=30, ge=1, le=480)


def _alert_headers(current_user: Any) -> dict[str, str]:
    return {
        "X-User-Id": current_user.user_id,
        "X-User-Role": current_user.role.value,
    }


@router.get("", summary="Role-filtered alert feed")
async def list_alerts(
    current_user: CurrentUser,
    ward_id: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    alert_type: str | None = Query(default=None),
    acknowledged: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return alerts filtered by the authenticated user's role and ward access."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if ward_id:
        params["ward_id"] = ward_id
    if priority:
        params["priority"] = priority
    if alert_type:
        params["alert_type"] = alert_type
    if acknowledged is not None:
        params["acknowledged"] = str(acknowledged).lower()

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.alert_service_url}/alerts",
                params=params,
                headers=_alert_headers(current_user),
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Alert service timeout")


@router.post("/{alert_id}/acknowledge", summary="Acknowledge alert")
async def acknowledge_alert(
    alert_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.alert_service_url}/alerts/{alert_id}/acknowledge",
                headers=_alert_headers(current_user),
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Alert service timeout")


@router.post("/{alert_id}/snooze", summary="Snooze alert")
async def snooze_alert(
    alert_id: str,
    body: SnoozeRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.alert_service_url}/alerts/{alert_id}/snooze",
                json=body.model_dump(),
                headers=_alert_headers(current_user),
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Alert service timeout")


@router.post("/{alert_id}/escalate", summary="Escalate alert")
async def escalate_alert(
    alert_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.alert_service_url}/alerts/{alert_id}/escalate",
                headers=_alert_headers(current_user),
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Alert service timeout")
