"""
OncoIDT Simulation Service — Simulation Execution Engine.

Executes what-if counterfactual simulations over PatientTwin and WardTwin
states in a fully sandboxed mode.  Never writes to live patient or ward state.

Single-patient simulation:
  1. Clone patient twin from DB
  2. Apply interventions in apply_at_hours order
  3. Run sandboxed inference (calls inference service with counterfactual features)
  4. Produce RiskScore trajectories at 6h, 12h, 24h, 48h horizons
  5. Return within 10 seconds; partial results with truncated=True on timeout

Ward-level simulation:
  1. Clone WardTwin (all active patients)
  2. Apply intervention to all affected patients
  3. Re-run inference for each patient
  4. Aggregate results

Requirements: 9.2, 9.3, 9.4
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.models import RiskScore, RuleOverride
from services.simulation.interventions import (
    ClonedPatientState,
    Intervention,
    apply_intervention,
    clone_patient_twin,
    export_feature_vector,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Forecast horizons for counterfactual trajectories
INFECTION_HORIZONS = [6, 12, 24, 48]
DETERIORATION_HORIZONS = [6, 12, 24]

# Single-patient simulation timeout (seconds)
SIMULATION_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualScores:
    """Risk score trajectories for a single patient under a counterfactual."""
    patient_id: str
    infection_risk: dict[str, float]        # horizon_label → score, e.g. "6h" → 0.42
    deterioration_risk: dict[str, float]
    infection_uncertainty: dict[str, tuple[float, float]]   # horizon → (lower, upper)
    deterioration_uncertainty: dict[str, tuple[float, float]]
    model_version: str
    feature_snapshot_id: str
    counterfactual: bool = True


@dataclass
class SimulationResult:
    """Full result of a simulation run."""
    simulation_id: str
    patient_id: str | None
    ward_id: str | None
    interventions: list[dict[str, Any]]
    baseline_scores: CounterfactualScores | None
    counterfactual_scores: CounterfactualScores | None
    # Per-patient results for ward-level simulations
    ward_patient_results: list[CounterfactualScores] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Sandboxed inference
# ---------------------------------------------------------------------------

async def _run_sandboxed_inference(
    features: dict[str, Any],
    patient_id: str,
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Call the inference service's on-demand scoring endpoint with
    counterfactual features injected via a special header/param.

    Falls back to a direct feature-based heuristic if the inference
    service is unavailable (graceful degradation).
    """
    try:
        url = f"{settings.inference_service_url}/inference/score/{patient_id}"
        resp = await http_client.post(
            url,
            json={"features": features, "counterfactual": True},
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(
            "Sandboxed inference call failed for patient=%s (%s); using heuristic fallback",
            patient_id, exc,
        )
        return _heuristic_inference(features)


def _heuristic_inference(features: dict[str, Any]) -> dict[str, Any]:
    """
    Simple heuristic fallback when the inference service is unavailable.
    Uses clinical rules to produce approximate risk scores.
    """
    anc = features.get("anc") or 2.0
    temp = features.get("temperature_c") or 37.0
    immuno = features.get("immunosuppression_score") or 0.3
    antibiotic_active = bool(features.get("antibiotic_active", False))
    co_located = int(features.get("co_located_active_infections") or 0)

    # Infection risk heuristic
    base_infection = 0.2
    if anc < 0.5:
        base_infection += 0.4
    elif anc < 1.0:
        base_infection += 0.2
    if temp > 38.3:
        base_infection += 0.2
    base_infection += immuno * 0.15
    base_infection += co_located * 0.05
    if antibiotic_active:
        base_infection *= 0.65
    base_infection = min(1.0, max(0.0, base_infection))

    # Deterioration risk heuristic
    base_deterioration = 0.15 + immuno * 0.1
    if temp > 38.5:
        base_deterioration += 0.15
    base_deterioration = min(1.0, max(0.0, base_deterioration))

    conformal_q = 0.1
    return {
        "infection_risk": [base_infection] * 4,
        "deterioration_risk": [base_deterioration] * 3,
        "survival_median_hours": 72.0,
        "survival_ci_lower": 48.0,
        "survival_ci_upper": 96.0,
        "conformal_quantile": conformal_q,
        "model_version": "heuristic-fallback",
        "feature_snapshot_id": str(uuid.uuid4()),
    }


def _build_counterfactual_scores(
    patient_id: str,
    inference_result: dict[str, Any],
) -> CounterfactualScores:
    """Convert raw inference result dict to CounterfactualScores."""
    infection_raw = inference_result.get("infection_risk", [0.5] * 4)
    deterioration_raw = inference_result.get("deterioration_risk", [0.5] * 3)
    conformal_q = float(inference_result.get("conformal_quantile", 0.1))
    model_version = inference_result.get("model_version", "unknown")
    snapshot_id = inference_result.get("feature_snapshot_id", str(uuid.uuid4()))

    infection_risk = {}
    infection_uncertainty = {}
    for i, h in enumerate(INFECTION_HORIZONS):
        score = float(infection_raw[i]) if i < len(infection_raw) else 0.5
        infection_risk[f"{h}h"] = round(score, 6)
        infection_uncertainty[f"{h}h"] = (
            round(max(0.0, score - conformal_q), 6),
            round(min(1.0, score + conformal_q), 6),
        )

    deterioration_risk = {}
    deterioration_uncertainty = {}
    for i, h in enumerate(DETERIORATION_HORIZONS):
        score = float(deterioration_raw[i]) if i < len(deterioration_raw) else 0.5
        deterioration_risk[f"{h}h"] = round(score, 6)
        deterioration_uncertainty[f"{h}h"] = (
            round(max(0.0, score - conformal_q), 6),
            round(min(1.0, score + conformal_q), 6),
        )

    return CounterfactualScores(
        patient_id=patient_id,
        infection_risk=infection_risk,
        deterioration_risk=deterioration_risk,
        infection_uncertainty=infection_uncertainty,
        deterioration_uncertainty=deterioration_uncertainty,
        model_version=model_version,
        feature_snapshot_id=snapshot_id,
    )


# ---------------------------------------------------------------------------
# Single-patient simulation
# ---------------------------------------------------------------------------

async def _simulate_patient(
    patient_id: str,
    interventions: list[Intervention],
    session: AsyncSession,
    http_client: httpx.AsyncClient,
) -> tuple[CounterfactualScores, CounterfactualScores]:
    """
    Run baseline + counterfactual inference for a single patient.
    Returns (baseline_scores, counterfactual_scores).
    """
    # Clone patient twin
    cloned_state: ClonedPatientState = await clone_patient_twin(patient_id, session)

    # --- Baseline: run inference on unmodified clone ---
    baseline_features = export_feature_vector(cloned_state)
    baseline_result = await _run_sandboxed_inference(baseline_features, patient_id, http_client)
    baseline_scores = _build_counterfactual_scores(patient_id, baseline_result)

    # --- Counterfactual: apply interventions in apply_at_hours order ---
    sorted_interventions = sorted(interventions, key=lambda i: i.apply_at_hours)
    for intervention in sorted_interventions:
        apply_intervention(cloned_state, intervention)

    cf_features = export_feature_vector(cloned_state)
    cf_result = await _run_sandboxed_inference(cf_features, patient_id, http_client)
    cf_scores = _build_counterfactual_scores(patient_id, cf_result)

    return baseline_scores, cf_scores


async def run_patient_simulation(
    simulation_id: str,
    patient_id: str,
    interventions: list[Intervention],
    session: AsyncSession,
) -> SimulationResult:
    """
    Execute a single-patient what-if simulation.

    Returns within SIMULATION_TIMEOUT_SECONDS; sets truncated=True if
    the timeout is reached before completion.

    Requirements: 9.2, 9.3
    """
    intervention_dicts = [
        {
            "type": iv.intervention_type,
            "parameters": iv.parameters,
            "apply_at_hours": iv.apply_at_hours,
        }
        for iv in interventions
    ]

    async with httpx.AsyncClient() as http_client:
        try:
            baseline, counterfactual = await asyncio.wait_for(
                _simulate_patient(patient_id, interventions, session, http_client),
                timeout=SIMULATION_TIMEOUT_SECONDS,
            )
            return SimulationResult(
                simulation_id=simulation_id,
                patient_id=patient_id,
                ward_id=None,
                interventions=intervention_dicts,
                baseline_scores=baseline,
                counterfactual_scores=counterfactual,
                truncated=False,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Simulation %s timed out after %ss for patient=%s",
                simulation_id, SIMULATION_TIMEOUT_SECONDS, patient_id,
            )
            return SimulationResult(
                simulation_id=simulation_id,
                patient_id=patient_id,
                ward_id=None,
                interventions=intervention_dicts,
                baseline_scores=None,
                counterfactual_scores=None,
                truncated=True,
                error="Simulation timed out",
            )
        except Exception as exc:
            logger.exception("Simulation %s failed for patient=%s", simulation_id, patient_id)
            return SimulationResult(
                simulation_id=simulation_id,
                patient_id=patient_id,
                ward_id=None,
                interventions=intervention_dicts,
                baseline_scores=None,
                counterfactual_scores=None,
                truncated=False,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Ward-level simulation
# ---------------------------------------------------------------------------

async def run_ward_simulation(
    simulation_id: str,
    ward_id: str,
    interventions: list[Intervention],
    session: AsyncSession,
) -> SimulationResult:
    """
    Execute a ward-level what-if simulation.

    Clones all active patients in the ward, applies the intervention to
    each, and re-runs inference.  Results are aggregated per patient.

    Requirements: 9.4
    """
    from sqlalchemy import text

    # Fetch all active patient IDs in the ward
    result = await session.execute(
        text(
            """
            SELECT DISTINCT patient_id
            FROM feature_snapshots
            WHERE features->>'ward_id' = :ward_id
              AND features->>'status' = 'active'
            ORDER BY patient_id
            """
        ),
        {"ward_id": ward_id},
    )
    patient_ids = [str(row[0]) for row in result.fetchall()]

    if not patient_ids:
        logger.warning("No active patients found for ward=%s", ward_id)
        return SimulationResult(
            simulation_id=simulation_id,
            patient_id=None,
            ward_id=ward_id,
            interventions=[
                {"type": iv.intervention_type, "parameters": iv.parameters,
                 "apply_at_hours": iv.apply_at_hours}
                for iv in interventions
            ],
            baseline_scores=None,
            counterfactual_scores=None,
            ward_patient_results=[],
        )

    patient_results: list[CounterfactualScores] = []
    truncated = False

    async with httpx.AsyncClient() as http_client:
        for patient_id in patient_ids:
            try:
                _, cf_scores = await asyncio.wait_for(
                    _simulate_patient(patient_id, interventions, session, http_client),
                    timeout=SIMULATION_TIMEOUT_SECONDS,
                )
                patient_results.append(cf_scores)
            except asyncio.TimeoutError:
                logger.warning(
                    "Ward simulation %s timed out for patient=%s",
                    simulation_id, patient_id,
                )
                truncated = True
                break
            except Exception:
                logger.exception(
                    "Ward simulation %s failed for patient=%s — skipping",
                    simulation_id, patient_id,
                )

    return SimulationResult(
        simulation_id=simulation_id,
        patient_id=None,
        ward_id=ward_id,
        interventions=[
            {"type": iv.intervention_type, "parameters": iv.parameters,
             "apply_at_hours": iv.apply_at_hours}
            for iv in interventions
        ],
        baseline_scores=None,
        counterfactual_scores=None,
        ward_patient_results=patient_results,
        truncated=truncated,
    )
