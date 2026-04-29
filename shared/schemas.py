"""
OncoIDT Pydantic v2 schemas — API serialization layer.
Mirrors shared/models.py dataclasses with full validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Supporting leaf schemas
# ---------------------------------------------------------------------------

class RuleOverrideSchema(BaseModel):
    rule_id: str
    threshold_value: float
    triggered_value: float
    score_floor: float


class DataQualityFlagSchema(BaseModel):
    flag_type: str
    field_name: str
    message: str
    severity: Literal["info", "warning", "error"]


class VitalsSnapshotSchema(BaseModel):
    temperature_c: float | None = None
    heart_rate_bpm: float | None = None
    respiratory_rate_rpm: float | None = None
    sbp_mmhg: float | None = None
    dbp_mmhg: float | None = None
    spo2_pct: float | None = None
    gcs: int | None = None
    timestamp: datetime


class LabSnapshotSchema(BaseModel):
    anc: float | None = None            # × 10⁹/L
    wbc: float | None = None
    lymphocytes: float | None = None
    crp_mg_l: float | None = None
    procalcitonin_ug_l: float | None = None
    lactate_mmol_l: float | None = None
    creatinine_umol_l: float | None = None
    timestamp: datetime


class RiskScoreSchema(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    uncertainty_lower: float = Field(ge=0.0, le=1.0)
    uncertainty_upper: float = Field(ge=0.0, le=1.0)
    forecast_horizon_hours: int = Field(gt=0)
    model_version: str
    feature_snapshot_id: str
    rule_overrides: list[RuleOverrideSchema] = Field(default_factory=list)
    timestamp: datetime


class SurvivalEstimateSchema(BaseModel):
    median_hours: float = Field(gt=0)
    ci_80_lower_hours: float = Field(gt=0)
    ci_80_upper_hours: float = Field(gt=0)
    event_type: Literal["infection", "deterioration", "icu_transfer"]
    model_version: str
    timestamp: datetime


class BedStateSchema(BaseModel):
    bed_id: str
    room_id: str
    zone: str
    patient_id: str | None = None
    infection_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    deterioration_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    exposure_flag: bool = False
    last_score_timestamp: datetime | None = None


class ExposureEventSchema(BaseModel):
    event_id: str
    source_patient_id: str
    pathogen: str | None = None
    affected_patient_ids: list[str] = Field(default_factory=list)
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)


class EnvironmentalContextSchema(BaseModel):
    air_quality_index: float | None = None
    temperature_celsius: float | None = None
    humidity_pct: float | None = None
    timestamp: datetime


# ---------------------------------------------------------------------------
# Canonical record payload schemas
# ---------------------------------------------------------------------------

class LabRecordSchema(BaseModel):
    loinc_code: str
    source_code: str
    source_system_code: str
    value_numeric: float | None = None
    value_text: str | None = None
    unit: str
    reference_range_low: float | None = None
    reference_range_high: float | None = None
    abnormal_flag: str | None = None


class VitalRecordSchema(BaseModel):
    loinc_code: str
    value_numeric: float | None = None
    value_text: str | None = None
    unit: str


class MedicationRecordSchema(BaseModel):
    rxnorm_code: str
    source_code: str
    drug_name: str
    dose_mg: float | None = None
    route: str
    is_chemotherapy: bool = False
    chemo_regimen_code: str | None = None
    administration_timestamp: datetime


class ObservationRecordSchema(BaseModel):
    observation_type: str
    value_text: str | None = None
    value_numeric: float | None = None
    unit: str | None = None


class ClinicalEventRecordSchema(BaseModel):
    event_type: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoteMetadataRecordSchema(BaseModel):
    note_type: str
    author_role: str
    word_count: int | None = None


# ---------------------------------------------------------------------------
# Canonical record wrapper schema
# ---------------------------------------------------------------------------

class CanonicalRecordSchema(BaseModel):
    record_id: str
    patient_id: str
    source_system: str
    source_record_id: str
    record_type: Literal["vital", "lab", "medication", "observation", "event", "note_metadata"]
    timestamp_utc: datetime
    ingested_at: datetime
    data_quality_flags: list[DataQualityFlagSchema] = Field(default_factory=list)
    payload: (
        VitalRecordSchema
        | LabRecordSchema
        | MedicationRecordSchema
        | ObservationRecordSchema
        | ClinicalEventRecordSchema
        | NoteMetadataRecordSchema
    )


# ---------------------------------------------------------------------------
# Top-level twin schemas
# ---------------------------------------------------------------------------

class PatientTwinSchema(BaseModel):
    # Identity
    patient_id: str
    mrn: str
    ward_id: str
    bed_id: str | None = None
    admission_timestamp: datetime
    discharge_timestamp: datetime | None = None
    status: Literal["active", "archived", "transferred"]

    # Demographics
    age_years: int = Field(ge=0, le=150)
    sex: Literal["M", "F", "O", "U"]
    primary_diagnosis_icd10: str
    comorbidities: list[str] = Field(default_factory=list)

    # Chemotherapy
    chemo_regimen: str
    chemo_cycle_number: int = Field(ge=0)
    chemo_cycle_phase: Literal["pre", "nadir", "recovery", "off"]
    days_since_last_chemo_dose: float = Field(ge=0.0)
    cumulative_dose_mg_m2: float = Field(ge=0.0)
    immunosuppression_score: float = Field(ge=0.0, le=1.0)

    # Current snapshots
    vitals: VitalsSnapshotSchema
    labs: LabSnapshotSchema

    # Risk scores
    infection_risk_scores: dict[str, RiskScoreSchema] = Field(default_factory=dict)
    deterioration_risk_scores: dict[str, RiskScoreSchema] = Field(default_factory=dict)
    survival_estimate: SurvivalEstimateSchema | None = None

    # Metadata
    last_updated: datetime
    data_quality_flags: list[DataQualityFlagSchema] = Field(default_factory=list)
    feature_version: str


class WardTwinSchema(BaseModel):
    ward_id: str
    ward_name: str
    total_beds: int = Field(ge=0)
    occupied_beds: int = Field(ge=0)
    last_updated: datetime

    beds: dict[str, BedStateSchema] = Field(default_factory=dict)

    ward_infection_risk: float = Field(ge=0.0, le=1.0)
    ward_deterioration_risk: float = Field(ge=0.0, le=1.0)
    high_risk_patient_count: int = Field(ge=0)

    environmental: EnvironmentalContextSchema | None = None

    active_exposure_events: list[ExposureEventSchema] = Field(default_factory=list)
    recent_confirmed_infections: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API response envelopes
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"] = "ok"
    service: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
