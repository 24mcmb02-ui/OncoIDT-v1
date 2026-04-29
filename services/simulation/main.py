"""
OncoIDT Simulation Service — FastAPI Application.

Exposes async job-based simulation endpoints:
  POST /simulations          — submit a SimulationRequest, returns {job_id}
  GET  /simulations/{job_id} — poll status and results
  GET  /health               — liveness probe
  GET  /ready                — readiness probe

All simulation outputs are labelled counterfactual=True and are NEVER
written to the live patient_twin or ward_twin state.

Simulation sessions are persisted to the simulation_sessions table with
requesting_user_id, input parameters, and output.

Requirements: 9.5, 9.6, 9.7
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from services.simulation.engine import (
    SimulationResult,
    run_patient_simulation,
    run_ward_simulation,
)
from services.simulation.interventions import Intervention, InterventionType
from shared.audit import append_audit_entry
from shared.auth import CurrentUser
from shared.config import get_settings
from shared.db import DbSession, close_engine, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, get_redis

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

# ---------------------------------------------------------------------------
# In-process job store (job_id → status/result)
# Suitable for single-instance deployment; replace with Redis for multi-instance.
# ---------------------------------------------------------------------------

JobStatus = Literal["pending", "running", "complete", "failed"]

_jobs: dict[str, dict[str, Any]] = {}


def _set_job(job_id: str, status: JobStatus, result: SimulationResult | None = None, error: str | None = None) -> None:
    _jobs[job_id] = {
        "job_id": job_id,
        "status": status,
        "result": result,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class InterventionRequest(BaseModel):
    intervention_type: InterventionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    apply_at_hours: float = Field(default=0.0, ge=0.0)


class SimulationRequest(BaseModel):
    patient_id: str | None = None
    ward_id: str | None = None
    interventions: list[InterventionRequest] = Field(..., min_length=1, max_length=5)
    horizons: list[int] = Field(default=[6, 12, 24, 48])

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "550e8400-e29b-41d4-a716-446655440000",
            "interventions": [
                {
                    "intervention_type": "antibiotic_administration",
                    "parameters": {"antibiotic_name": "piperacillin-tazobactam"},
                    "apply_at_hours": 0.0,
                }
            ],
        }
    }}


class SimulationJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = ""


class CounterfactualScoresResponse(BaseModel):
    patient_id: str
    infection_risk: dict[str, float]
    deterioration_risk: dict[str, float]
    infection_uncertainty: dict[str, list[float]]
    deterioration_uncertainty: dict[str, list[float]]
    model_version: str
    counterfactual: bool = True


class SimulationResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    simulation_id: str | None = None
    patient_id: str | None = None
    ward_id: str | None = None
    interventions: list[dict[str, Any]] = Field(default_factory=list)
    baseline_scores: CounterfactualScoresResponse | None = None
    counterfactual_scores: CounterfactualScoresResponse | None = None
    ward_patient_results: list[CounterfactualScoresResponse] = Field(default_factory=list)
    truncated: bool = False
    counterfactual: bool = True
    error: str | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _persist_simulation_session(
    session: Any,
    simulation_id: str,
    requesting_user_id: str,
    request_payload: dict[str, Any],
    result: SimulationResult,
) -> None:
    """Persist simulation session to simulation_sessions table."""
    await session.execute(
        text(
            """
            INSERT INTO simulation_sessions (
                session_id, requesting_user_id, patient_id, ward_id,
                input_parameters, output, truncated, counterfactual,
                created_at, completed_at
            ) VALUES (
                :session_id, :requesting_user_id, :patient_id, :ward_id,
                :input_parameters::jsonb, :output::jsonb, :truncated, :counterfactual,
                :created_at, :completed_at
            )
            """
        ),
        {
            "session_id": simulation_id,
            "requesting_user_id": requesting_user_id,
            "patient_id": result.patient_id,
            "ward_id": result.ward_id,
            "input_parameters": json.dumps(request_payload),
            "output": json.dumps(_result_to_dict(result)),
            "truncated": result.truncated,
            "counterfactual": True,
            "created_at": datetime.now(timezone.utc),
            "completed_at": result.completed_at,
        },
    )


def _result_to_dict(result: SimulationResult) -> dict[str, Any]:
    """Serialize SimulationResult to a JSON-safe dict."""
    def scores_to_dict(s: Any) -> dict[str, Any] | None:
        if s is None:
            return None
        return {
            "patient_id": s.patient_id,
            "infection_risk": s.infection_risk,
            "deterioration_risk": s.deterioration_risk,
            "infection_uncertainty": {
                k: list(v) for k, v in s.infection_uncertainty.items()
            },
            "deterioration_uncertainty": {
                k: list(v) for k, v in s.deterioration_uncertainty.items()
            },
            "model_version": s.model_version,
            "feature_snapshot_id": s.feature_snapshot_id,
            "counterfactual": s.counterfactual,
        }

    return {
        "simulation_id": result.simulation_id,
        "patient_id": result.patient_id,
        "ward_id": result.ward_id,
        "interventions": result.interventions,
        "baseline_scores": scores_to_dict(result.baseline_scores),
        "counterfactual_scores": scores_to_dict(result.counterfactual_scores),
        "ward_patient_results": [scores_to_dict(s) for s in result.ward_patient_results],
        "truncated": result.truncated,
        "error": result.error,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
    }


def _scores_to_response(s: Any) -> CounterfactualScoresResponse | None:
    if s is None:
        return None
    return CounterfactualScoresResponse(
        patient_id=s.patient_id,
        infection_risk=s.infection_risk,
        deterioration_risk=s.deterioration_risk,
        infection_uncertainty={k: list(v) for k, v in s.infection_uncertainty.items()},
        deterioration_uncertainty={k: list(v) for k, v in s.deterioration_uncertainty.items()},
        model_version=s.model_version,
        counterfactual=True,
    )


# ---------------------------------------------------------------------------
# Background simulation runner
# ---------------------------------------------------------------------------

async def _run_simulation_job(
    job_id: str,
    simulation_id: str,
    request: SimulationRequest,
    requesting_user_id: str,
) -> None:
    """
    Background task: execute simulation, persist session, update job store.
    Never writes to live patient_twin or ward_twin.
    """
    _set_job(job_id, "running")

    interventions = [
        Intervention(
            intervention_type=iv.intervention_type,
            parameters=iv.parameters,
            apply_at_hours=iv.apply_at_hours,
        )
        for iv in request.interventions
    ]

    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            if request.patient_id:
                result = await run_patient_simulation(
                    simulation_id=simulation_id,
                    patient_id=request.patient_id,
                    interventions=interventions,
                    session=session,
                )
            elif request.ward_id:
                result = await run_ward_simulation(
                    simulation_id=simulation_id,
                    ward_id=request.ward_id,
                    interventions=interventions,
                    session=session,
                )
            else:
                raise ValueError("Either patient_id or ward_id must be provided")

            # Persist session (counterfactual=True, never touches live state)
            request_payload = request.model_dump()
            await _persist_simulation_session(
                session, simulation_id, requesting_user_id, request_payload, result
            )

            # Audit log entry
            await append_audit_entry(
                session,
                user_id=requesting_user_id,
                action="simulation_completed",
                resource_type="simulation_session",
                resource_id=simulation_id,
                details={
                    "patient_id": result.patient_id,
                    "ward_id": result.ward_id,
                    "truncated": result.truncated,
                    "intervention_count": len(interventions),
                },
            )
            await session.commit()

            _set_job(job_id, "complete", result=result)
            logger.info(
                "Simulation %s completed (job=%s truncated=%s)",
                simulation_id, job_id, result.truncated,
            )

        except Exception as exc:
            logger.exception("Simulation job %s failed: %s", job_id, exc)
            await session.rollback()
            _set_job(job_id, "failed", error=str(exc))


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
        await get_redis().ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    logger.info("Simulation service started.")
    yield
    await graceful_shutdown([error_monitor_task])
    await close_engine()
    await close_redis_pool()
    logger.info("Simulation service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Simulation Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="simulation-service",
        readiness_checks={"db": _check_db, "redis": _check_redis},
    )
)

router = APIRouter(prefix="/simulations", tags=["simulations"])


# ---------------------------------------------------------------------------
# POST /simulations — submit simulation request
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=SimulationJobResponse,
    status_code=202,
    summary="Submit a what-if simulation request",
)
async def submit_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
) -> SimulationJobResponse:
    """
    Accept a SimulationRequest, enqueue an async job, and return {job_id}.

    Simulation outputs are always labelled counterfactual=True and are
    NEVER written to the live patient_twin or ward_twin state.

    Provide either patient_id (single-patient) or ward_id (ward-level).
    """
    if not request.patient_id and not request.ward_id:
        raise HTTPException(
            status_code=422,
            detail="Either patient_id or ward_id must be provided.",
        )

    job_id = str(uuid.uuid4())
    simulation_id = str(uuid.uuid4())

    _set_job(job_id, "pending")

    background_tasks.add_task(
        _run_simulation_job,
        job_id=job_id,
        simulation_id=simulation_id,
        request=request,
        requesting_user_id=current_user.user_id,
    )

    logger.info(
        "Simulation job %s submitted by user=%s patient=%s ward=%s",
        job_id, current_user.user_id, request.patient_id, request.ward_id,
    )

    return SimulationJobResponse(
        job_id=job_id,
        status="pending",
        message="Simulation job submitted. Poll GET /simulations/{job_id} for results.",
    )


# ---------------------------------------------------------------------------
# GET /simulations/{job_id} — poll status and results
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}",
    response_model=SimulationResultResponse,
    summary="Poll simulation job status and results",
)
async def get_simulation(
    job_id: str,
    current_user: CurrentUser,
) -> SimulationResultResponse:
    """
    Return the current status and results of a simulation job.

    Status values: pending | running | complete | failed

    All results carry counterfactual=True — they represent hypothetical
    trajectories and must not be interpreted as live clinical data.
    """
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Simulation job {job_id} not found.")

    status: JobStatus = job["status"]
    result: SimulationResult | None = job.get("result")
    error: str | None = job.get("error")

    if status in ("pending", "running"):
        return SimulationResultResponse(
            job_id=job_id,
            status=status,
            counterfactual=True,
        )

    if status == "failed":
        return SimulationResultResponse(
            job_id=job_id,
            status="failed",
            error=error,
            counterfactual=True,
        )

    # status == "complete"
    assert result is not None

    ward_results = [
        _scores_to_response(s)
        for s in result.ward_patient_results
        if s is not None
    ]

    return SimulationResultResponse(
        job_id=job_id,
        status="complete",
        simulation_id=result.simulation_id,
        patient_id=result.patient_id,
        ward_id=result.ward_id,
        interventions=result.interventions,
        baseline_scores=_scores_to_response(result.baseline_scores),
        counterfactual_scores=_scores_to_response(result.counterfactual_scores),
        ward_patient_results=[r for r in ward_results if r is not None],
        truncated=result.truncated,
        counterfactual=True,
        error=result.error,
        completed_at=result.completed_at,
    )


app.include_router(router)
