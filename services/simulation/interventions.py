"""
OncoIDT Simulation Service — Patient Twin Cloning and Intervention Application.

Provides:
  - ``clone_patient_twin(patient_id, session)`` — deep copy of current twin state,
    never modifies live state.
  - ``apply_intervention(twin, intervention)`` — apply a single intervention as a
    structural state mutation on the cloned PatientTwin feature vector.

Intervention types and causal effect models:
  - ``antibiotic_administration``: set antibiotic_active=True, apply time-decay
    function on infection risk trajectory.
  - ``dose_modification``: update chemo_cycle_phase, days_since_last_chemo_dose,
    recompute immunosuppression_score.
  - ``isolation_measure``: remove CO_LOCATED edges in cloned graph state (reflected
    as zeroing co_located_active_infections feature).

Requirements: 9.1, 9.5, 22.4
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.models import PatientTwin, VitalsSnapshot, LabSnapshot, RiskScore, SurvivalEstimate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intervention data model
# ---------------------------------------------------------------------------

InterventionType = Literal[
    "antibiotic_administration",
    "dose_modification",
    "isolation_measure",
]


@dataclass
class Intervention:
    """A single what-if intervention to apply to a cloned patient twin."""
    intervention_type: InterventionType
    # Intervention-specific parameters
    parameters: dict[str, Any] = field(default_factory=dict)
    # Hours from simulation start at which to apply this intervention
    apply_at_hours: float = 0.0


@dataclass
class ClonedPatientState:
    """
    A sandboxed copy of a PatientTwin with additional simulation metadata.
    Never references or modifies the live PatientTwin.
    """
    twin: PatientTwin
    # Derived feature overrides applied by interventions (flat dict)
    feature_overrides: dict[str, Any] = field(default_factory=dict)
    # Graph state overrides (e.g., co-location edges removed by isolation)
    graph_overrides: dict[str, Any] = field(default_factory=dict)
    # Audit trail of applied interventions
    applied_interventions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Immunosuppression score computation
# ---------------------------------------------------------------------------

# Regimen-specific nadir severity weights (higher = more immunosuppressive)
_REGIMEN_WEIGHTS: dict[str, float] = {
    "R-CHOP": 0.75,
    "BEP": 0.80,
    "FOLFOX": 0.55,
    "ABVD": 0.65,
    "other": 0.60,
}

# Cycle phase multipliers
_PHASE_MULTIPLIERS: dict[str, float] = {
    "nadir": 1.0,
    "pre": 0.6,
    "recovery": 0.4,
    "off": 0.1,
}


def _compute_immunosuppression_score(
    chemo_regimen: str,
    chemo_cycle_phase: str,
    days_since_last_chemo_dose: float,
    cumulative_dose_mg_m2: float,
) -> float:
    """
    Compute a composite immunosuppression score in [0, 1].

    Model:
      base = regimen_weight × phase_multiplier
      time_decay = exp(-days_since_last_dose / 14)   # 14-day half-life
      cumulative_factor = min(1.0, cumulative_dose / 5000)
      score = base × time_decay × (0.7 + 0.3 × cumulative_factor)
    """
    regimen_weight = _REGIMEN_WEIGHTS.get(chemo_regimen, _REGIMEN_WEIGHTS["other"])
    phase_mult = _PHASE_MULTIPLIERS.get(chemo_cycle_phase, 0.5)
    time_decay = math.exp(-max(0.0, days_since_last_chemo_dose) / 14.0)
    cumulative_factor = min(1.0, cumulative_dose_mg_m2 / 5000.0)
    score = regimen_weight * phase_mult * time_decay * (0.7 + 0.3 * cumulative_factor)
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------

async def clone_patient_twin(
    patient_id: str,
    session: AsyncSession,
) -> ClonedPatientState:
    """
    Deep-copy the current PatientTwin state from the database.

    Reconstructs a PatientTwin from the latest risk_scores and
    feature_snapshots rows.  The returned ClonedPatientState is fully
    independent of the live state — mutations never propagate back.

    Raises ValueError if no active patient record is found.
    """
    # Fetch latest feature snapshot
    result = await session.execute(
        text(
            """
            SELECT features, feature_version, timestamp
            FROM feature_snapshots
            WHERE patient_id = :patient_id
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ),
        {"patient_id": patient_id},
    )
    snapshot_row = result.mappings().fetchone()

    # Fetch latest risk scores
    scores_result = await session.execute(
        text(
            """
            SELECT score_type, forecast_horizon_hours, score,
                   uncertainty_lower, uncertainty_upper, model_version,
                   feature_snapshot_id, rule_overrides, timestamp
            FROM risk_scores
            WHERE patient_id = :patient_id
            ORDER BY timestamp DESC
            LIMIT 20
            """
        ),
        {"patient_id": patient_id},
    )
    score_rows = scores_result.mappings().fetchall()

    import json as _json

    # Build risk score dicts
    infection_scores: dict[str, RiskScore] = {}
    deterioration_scores: dict[str, RiskScore] = {}
    for row in score_rows:
        rs = RiskScore(
            score=float(row["score"]),
            uncertainty_lower=float(row["uncertainty_lower"] or 0.0),
            uncertainty_upper=float(row["uncertainty_upper"] or 1.0),
            forecast_horizon_hours=int(row["forecast_horizon_hours"]),
            model_version=row["model_version"],
            feature_snapshot_id=str(row["feature_snapshot_id"]),
            rule_overrides=[],
            timestamp=row["timestamp"],
        )
        key = f"{row['forecast_horizon_hours']}h"
        if row["score_type"] == "infection":
            infection_scores.setdefault(key, rs)
        elif row["score_type"] == "deterioration":
            deterioration_scores.setdefault(key, rs)

    # Extract features for twin reconstruction
    features: dict[str, Any] = {}
    feature_version = "v1"
    if snapshot_row:
        raw = snapshot_row["features"]
        features = _json.loads(raw) if isinstance(raw, str) else (raw or {})
        feature_version = snapshot_row["feature_version"] or "v1"

    # Build a minimal PatientTwin from available data
    now = datetime.now(timezone.utc)
    twin = PatientTwin(
        patient_id=patient_id,
        mrn=features.get("mrn", "UNKNOWN"),
        ward_id=features.get("ward_id", "UNKNOWN"),
        bed_id=features.get("bed_id"),
        admission_timestamp=features.get("admission_timestamp", now),
        discharge_timestamp=None,
        status="active",
        age_years=int(features.get("age_years", 0)),
        sex=features.get("sex", "U"),
        primary_diagnosis_icd10=features.get("primary_diagnosis_icd10", "C00"),
        comorbidities=features.get("comorbidities", []),
        chemo_regimen=features.get("chemo_regimen", "other"),
        chemo_cycle_number=int(features.get("chemo_cycle_number", 1)),
        chemo_cycle_phase=features.get("chemo_cycle_phase", "off"),
        days_since_last_chemo_dose=float(features.get("days_since_last_chemo_dose", 0.0)),
        cumulative_dose_mg_m2=float(features.get("cumulative_dose_mg_m2", 0.0)),
        immunosuppression_score=float(features.get("immunosuppression_score", 0.0)),
        vitals=VitalsSnapshot(
            temperature_c=features.get("temperature_c"),
            heart_rate_bpm=features.get("heart_rate_bpm"),
            respiratory_rate_rpm=features.get("respiratory_rate_rpm"),
            sbp_mmhg=features.get("sbp_mmhg"),
            dbp_mmhg=features.get("dbp_mmhg"),
            spo2_pct=features.get("spo2_pct"),
            gcs=features.get("gcs"),
            timestamp=now,
        ),
        labs=LabSnapshot(
            anc=features.get("anc"),
            wbc=features.get("wbc"),
            lymphocytes=features.get("lymphocytes"),
            crp_mg_l=features.get("crp_mg_l"),
            procalcitonin_ug_l=features.get("procalcitonin_ug_l"),
            lactate_mmol_l=features.get("lactate_mmol_l"),
            creatinine_umol_l=features.get("creatinine_umol_l"),
            timestamp=now,
        ),
        infection_risk_scores=infection_scores,
        deterioration_risk_scores=deterioration_scores,
        survival_estimate=None,
        last_updated=now,
        data_quality_flags=[],
        feature_version=feature_version,
    )

    # Deep copy to ensure full isolation from any shared references
    cloned_twin = copy.deepcopy(twin)
    cloned_features = copy.deepcopy(features)

    return ClonedPatientState(
        twin=cloned_twin,
        feature_overrides=cloned_features,
    )


# ---------------------------------------------------------------------------
# Intervention application
# ---------------------------------------------------------------------------

def apply_intervention(
    state: ClonedPatientState,
    intervention: Intervention,
) -> ClonedPatientState:
    """
    Apply a single intervention as a structural state mutation on the
    cloned PatientTwin feature vector.

    Returns the same ClonedPatientState (mutated in place) for chaining.
    Never touches live state.
    """
    itype = intervention.intervention_type
    params = intervention.parameters

    if itype == "antibiotic_administration":
        _apply_antibiotic(state, params, intervention.apply_at_hours)
    elif itype == "dose_modification":
        _apply_dose_modification(state, params)
    elif itype == "isolation_measure":
        _apply_isolation(state, params)
    else:
        logger.warning("Unknown intervention type: %s — skipping", itype)
        return state

    state.applied_interventions.append(
        f"{itype}@{intervention.apply_at_hours}h"
    )
    logger.debug(
        "Applied intervention %s to patient %s",
        itype,
        state.twin.patient_id,
    )
    return state


def _apply_antibiotic(
    state: ClonedPatientState,
    params: dict[str, Any],
    apply_at_hours: float,
) -> None:
    """
    Antibiotic administration causal effect model.

    Sets antibiotic_active=True and applies a time-decay function to
    reduce the infection risk trajectory.

    Decay model: risk_reduction = max_reduction × (1 - exp(-apply_at_hours / tau))
    where tau = 6h (antibiotic onset half-life) and max_reduction = 0.35.
    """
    state.feature_overrides["antibiotic_active"] = True
    state.feature_overrides["time_since_last_antibiotic_hours"] = 0.0

    # Antibiotic name (informational)
    antibiotic_name = params.get("antibiotic_name", "broad_spectrum")
    state.feature_overrides["antibiotic_name"] = antibiotic_name

    # Time-decay risk reduction applied to infection scores
    tau = params.get("onset_tau_hours", 6.0)
    max_reduction = params.get("max_risk_reduction", 0.35)
    risk_reduction = max_reduction * (1.0 - math.exp(-apply_at_hours / tau))

    # Mutate cloned infection risk scores
    for key, rs in state.twin.infection_risk_scores.items():
        new_score = max(0.0, rs.score - risk_reduction)
        state.twin.infection_risk_scores[key] = RiskScore(
            score=round(new_score, 6),
            uncertainty_lower=max(0.0, rs.uncertainty_lower - risk_reduction * 0.8),
            uncertainty_upper=max(0.0, rs.uncertainty_upper - risk_reduction * 0.5),
            forecast_horizon_hours=rs.forecast_horizon_hours,
            model_version=rs.model_version,
            feature_snapshot_id=rs.feature_snapshot_id,
            rule_overrides=rs.rule_overrides,
            timestamp=rs.timestamp,
        )

    state.feature_overrides["_antibiotic_risk_reduction"] = round(risk_reduction, 4)


def _apply_dose_modification(
    state: ClonedPatientState,
    params: dict[str, Any],
) -> None:
    """
    Dose modification causal effect model.

    Updates chemo_cycle_phase, days_since_last_chemo_dose, and
    recomputes immunosuppression_score from the new state.
    """
    twin = state.twin

    if "chemo_cycle_phase" in params:
        new_phase = params["chemo_cycle_phase"]
        twin.chemo_cycle_phase = new_phase
        state.feature_overrides["chemo_cycle_phase"] = new_phase

    if "days_since_last_chemo_dose" in params:
        new_days = float(params["days_since_last_chemo_dose"])
        twin.days_since_last_chemo_dose = new_days
        state.feature_overrides["days_since_last_chemo_dose"] = new_days

    if "dose_reduction_pct" in params:
        # Reduce cumulative dose proportionally
        reduction = float(params["dose_reduction_pct"]) / 100.0
        twin.cumulative_dose_mg_m2 = twin.cumulative_dose_mg_m2 * (1.0 - reduction)
        state.feature_overrides["cumulative_dose_mg_m2"] = twin.cumulative_dose_mg_m2

    # Recompute immunosuppression score from updated state
    new_immuno = _compute_immunosuppression_score(
        twin.chemo_regimen,
        twin.chemo_cycle_phase,
        twin.days_since_last_chemo_dose,
        twin.cumulative_dose_mg_m2,
    )
    twin.immunosuppression_score = new_immuno
    state.feature_overrides["immunosuppression_score"] = new_immuno


def _apply_isolation(
    state: ClonedPatientState,
    params: dict[str, Any],
) -> None:
    """
    Isolation measure causal effect model.

    Removes CO_LOCATED edges in the cloned graph state by zeroing
    co_located_active_infections and optionally staff_contact_count_24h.
    """
    state.feature_overrides["co_located_active_infections"] = 0
    state.graph_overrides["co_located_edges_removed"] = True

    if params.get("full_isolation", False):
        # Full isolation also removes staff contacts
        state.feature_overrides["staff_contact_count_24h"] = 0
        state.graph_overrides["staff_contact_edges_removed"] = True

    logger.debug(
        "Isolation applied to patient %s — co-location edges cleared",
        state.twin.patient_id,
    )


# ---------------------------------------------------------------------------
# Feature vector export (for sandboxed inference)
# ---------------------------------------------------------------------------

def export_feature_vector(state: ClonedPatientState) -> dict[str, Any]:
    """
    Merge the base feature snapshot with all intervention overrides,
    returning a flat feature dict suitable for model inference.
    """
    merged = dict(state.feature_overrides)

    # Ensure key twin fields are reflected in the feature vector
    twin = state.twin
    merged.update({
        "chemo_cycle_phase": twin.chemo_cycle_phase,
        "chemo_cycle_number": twin.chemo_cycle_number,
        "days_since_last_chemo_dose": twin.days_since_last_chemo_dose,
        "cumulative_dose_mg_m2": twin.cumulative_dose_mg_m2,
        "immunosuppression_score": twin.immunosuppression_score,
        "temperature_c": twin.vitals.temperature_c,
        "heart_rate_bpm": twin.vitals.heart_rate_bpm,
        "respiratory_rate_rpm": twin.vitals.respiratory_rate_rpm,
        "sbp_mmhg": twin.vitals.sbp_mmhg,
        "dbp_mmhg": twin.vitals.dbp_mmhg,
        "spo2_pct": twin.vitals.spo2_pct,
        "gcs": twin.vitals.gcs,
        "anc": twin.labs.anc,
        "wbc": twin.labs.wbc,
        "lymphocytes": twin.labs.lymphocytes,
        "crp_mg_l": twin.labs.crp_mg_l,
        "procalcitonin_ug_l": twin.labs.procalcitonin_ug_l,
        "_feature_version": twin.feature_version,
        "_counterfactual": True,
    })
    return merged
