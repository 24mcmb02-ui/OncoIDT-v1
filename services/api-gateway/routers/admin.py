"""
OncoIDT API Gateway — Admin endpoints.

GET  /api/v1/admin/rules              — list active clinical rules
PUT  /api/v1/admin/rules/{id}         — update rule (admin only)
POST /api/v1/admin/models/promote/{id} — promote model version (admin only)

Requirements: 13.1, 8.3, 14.4
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from shared.auth import Role, TokenPayload, require_role
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

AdminUser = TokenPayload  # type alias for clarity


class RuleUpdateRequest(BaseModel):
    enabled: bool | None = None
    threshold: float | None = None
    weight: float | None = None
    description: str | None = None


@router.get("/rules", summary="List active clinical rules")
async def list_rules(
    current_user: TokenPayload = Depends(require_role(
        Role.SYSTEM_ADMINISTRATOR,
        Role.INFECTION_CONTROL_OFFICER,
        Role.CLINICIAN,
        Role.CHARGE_NURSE,
    )),
) -> dict[str, Any]:
    """Return all active clinical rules from the reasoner service."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.reasoner_service_url}/reasoner/rules",
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Reasoner service timeout")


@router.put("/rules/{rule_id}", summary="Update clinical rule (admin only)")
async def update_rule(
    rule_id: str,
    body: RuleUpdateRequest,
    current_user: TokenPayload = Depends(require_role(Role.SYSTEM_ADMINISTRATOR)),
) -> dict[str, Any]:
    """Update a clinical rule. Requires System_Administrator role."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{settings.reasoner_service_url}/reasoner/rules/{rule_id}",
                json=body.model_dump(exclude_none=True),
                headers={"X-User-Id": current_user.user_id},
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Reasoner service timeout")


@router.post("/models/promote/{version}", summary="Promote model version (admin only)")
async def promote_model(
    version: str,
    current_user: TokenPayload = Depends(require_role(Role.SYSTEM_ADMINISTRATOR)),
) -> dict[str, Any]:
    """
    Flag a model version as production candidate.
    Requires explicit human approval (System_Administrator role).
    Proxies to training service.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.training_service_url}/training/models/{version}/promote",
                headers={"X-User-Id": current_user.user_id},
                timeout=10.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Training service timeout")
