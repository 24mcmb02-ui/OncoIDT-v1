"""
OncoIDT API Gateway — Simulation proxy endpoints.

POST /api/v1/simulations           — submit simulation request (async job)
GET  /api/v1/simulations/{job_id}  — poll simulation result

Proxies to simulation-service. Simulations exceeding 10s return partial
results asynchronously via job polling (Requirement 13.8).

Requirements: 13.1, 13.8
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.auth import CurrentUser
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/simulations", tags=["simulations"])


class InterventionRequest(BaseModel):
    intervention_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    apply_at_hours: float = 0.0


class SimulationRequest(BaseModel):
    patient_id: str | None = None
    ward_id: str | None = None
    interventions: list[InterventionRequest] = Field(..., min_length=1, max_length=5)
    horizons: list[int] = Field(default=[6, 12, 24, 48])


@router.post("", status_code=202, summary="Submit what-if simulation")
async def submit_simulation(
    body: SimulationRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Submit a simulation request. Returns {job_id} immediately.
    Poll GET /api/v1/simulations/{job_id} for results.
    All outputs are labelled counterfactual=True.
    """
    if not body.patient_id and not body.ward_id:
        raise HTTPException(status_code=422, detail="Either patient_id or ward_id is required")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.simulation_service_url}/simulations",
                json=body.model_dump(),
                headers={"X-User-Id": current_user.user_id, "X-User-Role": current_user.role.value},
                timeout=10.0,
            )
        if resp.status_code in (200, 202):
            return resp.json()
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Simulation service timeout")


@router.get("/{job_id}", summary="Poll simulation result")
async def get_simulation(
    job_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Poll the status and results of a simulation job."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.simulation_service_url}/simulations/{job_id}",
                timeout=5.0,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Simulation job {job_id} not found")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Simulation service timeout")
