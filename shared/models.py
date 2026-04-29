"""
OncoIDT shared dataclasses — canonical in-memory data models.
All services import from here to ensure type consistency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# ---------------------------------------------------------------------------
# Supporting leaf models
# ---------------------------------------------------------------------------

@dataclass
class RuleOverride:
    rule_id: str
    threshold_value: float
    triggered_value: float
    score_floor: float


@dataclass
class DataQualityFlag:
    flag_type: str          # e.g. "missing", "out_of_range", "near_duplicate"
    field_name: str
    message: str
    severity: Literal["info", "warning", "error"]


@dataclass
class VitalsSnapshot:
    temperature_c: float | None
    heart_rate_bpm: float | None
    respiratory_rate_rpm: float | None
    sbp_mmhg: float | None
    dbp_mmhg: float | None
    spo2_pct: float | None
    gcs: int | None
    timestamp: datetime


@dataclass
class LabSnapshot:
    anc: float | None                   # × 10⁹/L
    wbc: float | None
    lymphocytes: float | None
    crp_mg_l: float | None
    procalcitonin_ug_l: float | None
    lactate_mmol_l: float | None
    creatinine_umol_l: float | None
    timestamp: datetime


@dataclass
class RiskScore:
    score: float                        # [0, 1]
    uncertainty_lower: float
    uncertainty_upper: float
    forecast_horizon_hours: int
    model_version: str
    feature_snapshot_id: str
    rule_overrides: list[RuleOverride]
    timestamp: datetime


@dataclass
class SurvivalEstimate:
    median_hours: float
    ci_80_lower_hours: float
    ci_80_upper_hours: float
    event_type: Literal["infection", "deterioration", "icu_transfer"]
    model_version: str
    timestamp: datetime


@dataclass
class BedState:
    bed_id: str
    room_id: str
    zone: str
    patient_id: str | None
    infection_risk_score: float | None
    deterioration_risk_score: float | None
    exposure_flag: bool
    last_score_timestamp: datetime | None


@dataclass
class ExposureEvent:
    event_id: str
    source_patient_id: str
    pathogen: str | None
    affected_patient_ids: list[str]
    timestamp: datetime
    confidence: float


@dataclass
class EnvironmentalContext:
    air_quality_index: float | None
    temperature_celsius: float | None
    humidity_pct: float | None
    timestamp: datetime


# ---------------------------------------------------------------------------
# Canonical record payload types
# ---------------------------------------------------------------------------

@dataclass
class LabRecord:
    loinc_code: str
    source_code: str
    source_system_code: str
    value_numeric: float | None
    value_text: str | None
    unit: str
    reference_range_low: float | None
    reference_range_high: float | None
    abnormal_flag: str | None


@dataclass
class VitalRecord:
    loinc_code: str
    value_numeric: float | None
    value_text: str | None
    unit: str


@dataclass
class MedicationRecord:
    rxnorm_code: str
    source_code: str
    drug_name: str
    dose_mg: float | None
    route: str
    is_chemotherapy: bool
    chemo_regimen_code: str | None
    administration_timestamp: datetime


@dataclass
class ObservationRecord:
    observation_type: str
    value_text: str | None
    value_numeric: float | None
    unit: str | None


@dataclass
class ClinicalEventRecord:
    event_type: str          # "admission", "discharge", "transfer", "infection_confirmed", etc.
    description: str | None
    metadata: dict           # type: ignore[type-arg]


@dataclass
class NoteMetadataRecord:
    note_type: str
    author_role: str
    word_count: int | None


# ---------------------------------------------------------------------------
# Canonical record wrapper
# ---------------------------------------------------------------------------

@dataclass
class CanonicalRecord:
    record_id: str           # UUID
    patient_id: str
    source_system: str
    source_record_id: str
    record_type: Literal["vital", "lab", "medication", "observation", "event", "note_metadata"]
    timestamp_utc: datetime
    ingested_at: datetime
    data_quality_flags: list[DataQualityFlag]
    payload: (
        VitalRecord
        | LabRecord
        | MedicationRecord
        | ObservationRecord
        | ClinicalEventRecord
        | NoteMetadataRecord
    )


# ---------------------------------------------------------------------------
# Top-level twin models
# ---------------------------------------------------------------------------

@dataclass
class PatientTwin:
    # Identity
    patient_id: str                     # UUID, internal
    mrn: str                            # Hospital MRN (encrypted at rest)
    ward_id: str
    bed_id: str | None
    admission_timestamp: datetime
    discharge_timestamp: datetime | None
    status: Literal["active", "archived", "transferred"]

    # Demographics
    age_years: int
    sex: Literal["M", "F", "O", "U"]
    primary_diagnosis_icd10: str
    comorbidities: list[str]            # ICD-10 codes

    # Chemotherapy (first-class, Requirement 1.5)
    chemo_regimen: str
    chemo_cycle_number: int
    chemo_cycle_phase: Literal["pre", "nadir", "recovery", "off"]
    days_since_last_chemo_dose: float
    cumulative_dose_mg_m2: float
    immunosuppression_score: float      # derived composite [0, 1]

    # Current snapshots
    vitals: VitalsSnapshot
    labs: LabSnapshot

    # Risk scores (current)
    infection_risk_scores: dict[str, RiskScore]       # keyed by forecast horizon label
    deterioration_risk_scores: dict[str, RiskScore]
    survival_estimate: SurvivalEstimate | None

    # Metadata
    last_updated: datetime
    data_quality_flags: list[DataQualityFlag]
    feature_version: str


@dataclass
class WardTwin:
    ward_id: str
    ward_name: str
    total_beds: int
    occupied_beds: int
    last_updated: datetime

    # Per-bed state
    beds: dict[str, BedState]

    # Aggregate risk
    ward_infection_risk: float          # mean of active patient scores
    ward_deterioration_risk: float
    high_risk_patient_count: int        # score > 0.6

    # Environmental context
    environmental: EnvironmentalContext | None

    # Exposure tracking
    active_exposure_events: list[ExposureEvent]
    recent_confirmed_infections: list[str]   # patient_ids, last 72h
