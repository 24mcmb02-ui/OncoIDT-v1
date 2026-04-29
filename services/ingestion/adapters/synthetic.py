"""
SyntheticAdapter — generate clinically realistic oncology patient cohorts.

ANC trajectory: piecewise (exponential decay → Gaussian nadir → logistic recovery)
Fever onset: Hawkes process with infection-event intensity increase in preceding 12–24h
Vitals: physiologically correlated Gaussian noise
Output: CanonicalRecord objects identical in format to real ingested data
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from shared.models import (
    CanonicalRecord,
    ClinicalEventRecord,
    DataQualityFlag,
    LabRecord,
    MedicationRecord,
    VitalRecord,
)
from services.ingestion.adapters.base import SourceAdapter, ValidationResult

# ---------------------------------------------------------------------------
# Regimen-specific ANC trajectory parameters
# ---------------------------------------------------------------------------

@dataclass
class RegimenParams:
    """ANC trajectory parameters per chemotherapy regimen."""
    baseline_anc: float          # × 10⁹/L, pre-chemo
    decay_rate: float            # exponential decay constant (per day)
    nadir_value: float           # mean ANC at nadir × 10⁹/L
    nadir_noise_std: float       # Gaussian noise std at nadir
    nadir_duration_days: float   # mean nadir duration
    recovery_rate: float         # logistic recovery steepness
    recovery_midpoint_days: float  # days post-nadir to 50% recovery


REGIMEN_PARAMS: dict[str, RegimenParams] = {
    "R-CHOP": RegimenParams(
        baseline_anc=4.5, decay_rate=0.6, nadir_value=0.4,
        nadir_noise_std=0.15, nadir_duration_days=5.0,
        recovery_rate=0.8, recovery_midpoint_days=4.0,
    ),
    "BEP": RegimenParams(
        baseline_anc=4.0, decay_rate=0.5, nadir_value=0.5,
        nadir_noise_std=0.12, nadir_duration_days=4.0,
        recovery_rate=0.7, recovery_midpoint_days=5.0,
    ),
    "FOLFOX": RegimenParams(
        baseline_anc=4.2, decay_rate=0.35, nadir_value=1.2,
        nadir_noise_std=0.2, nadir_duration_days=3.0,
        recovery_rate=0.6, recovery_midpoint_days=4.5,
    ),
    "other": RegimenParams(
        baseline_anc=4.0, decay_rate=0.45, nadir_value=0.8,
        nadir_noise_std=0.18, nadir_duration_days=4.5,
        recovery_rate=0.65, recovery_midpoint_days=4.5,
    ),
}

# LOINC codes for key lab/vital values
_LOINC_ANC = "26499-4"
_LOINC_WBC = "6690-2"
_LOINC_TEMP = "8310-5"
_LOINC_HR = "8867-4"
_LOINC_RR = "9279-1"
_LOINC_SBP = "8480-6"
_LOINC_DBP = "8462-4"
_LOINC_SPO2 = "2708-6"
_LOINC_GCS = "9269-2"
_LOINC_CRP = "1988-5"
_LOINC_PCT = "33959-8"


# ---------------------------------------------------------------------------
# ANC trajectory
# ---------------------------------------------------------------------------

def anc_at_day(day: float, params: RegimenParams, rng: np.random.Generator) -> float:
    """
    Compute ANC (× 10⁹/L) at `day` days post-chemo-dose.

    Phases:
      0 → nadir_start: exponential decay
      nadir_start → nadir_end: Gaussian noise around nadir_value
      nadir_end → ...: logistic recovery toward baseline
    """
    nadir_start = -math.log(params.nadir_value / params.baseline_anc) / params.decay_rate
    nadir_end = nadir_start + params.nadir_duration_days

    if day < nadir_start:
        # Pre-nadir: exponential decay
        anc = params.baseline_anc * math.exp(-params.decay_rate * day)
    elif day < nadir_end:
        # Nadir: Gaussian noise around nadir value
        anc = float(rng.normal(params.nadir_value, params.nadir_noise_std))
    else:
        # Recovery: logistic curve from nadir toward baseline
        days_post_nadir = day - nadir_end
        recovery_frac = 1.0 / (
            1.0 + math.exp(-params.recovery_rate * (days_post_nadir - params.recovery_midpoint_days))
        )
        anc = params.nadir_value + (params.baseline_anc - params.nadir_value) * recovery_frac

    return max(0.0, anc + float(rng.normal(0, 0.05)))


# ---------------------------------------------------------------------------
# Hawkes process for fever onset
# ---------------------------------------------------------------------------

def simulate_hawkes_fever_times(
    infection_event_times: list[float],
    horizon_days: float,
    baseline_intensity: float,
    excitation_amplitude: float,
    decay_rate: float,
    rng: np.random.Generator,
) -> list[float]:
    """
    Simulate fever onset times using a Hawkes process.

    Infection events increase fever intensity in the preceding 12–24h window
    (modeled as a backward-looking excitation: fever times cluster before infection).

    Returns list of fever onset times (days from admission).
    """
    # Build intensity function: baseline + sum of excitation kernels
    # Each infection event at t_inf excites fever in [t_inf - 24h, t_inf - 12h]
    fever_times: list[float] = []
    t = 0.0
    dt = 1.0 / 24.0  # 1-hour steps

    while t < horizon_days:
        # Compute intensity at time t
        intensity = baseline_intensity
        for t_inf in infection_event_times:
            # Excitation window: 12–24h before infection
            window_start = t_inf - 1.0   # 24h before
            window_end = t_inf - 0.5     # 12h before
            if window_start <= t <= window_end:
                intensity += excitation_amplitude * math.exp(
                    -decay_rate * (t - window_start)
                )

        # Thinning: accept with probability intensity * dt
        if rng.random() < intensity * dt:
            fever_times.append(t)

        t += dt

    return fever_times


# ---------------------------------------------------------------------------
# Vitals generation with physiological correlations
# ---------------------------------------------------------------------------

def generate_vitals_at_time(
    t_days: float,
    fever_times: list[float],
    anc: float,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    """
    Generate a vitals snapshot with physiologically correlated noise.

    Fever (temp > 38.3°C) drives correlated increases in HR and RR.
    Low ANC (< 0.5) slightly elevates baseline HR.
    """
    # Check if within 2h of a fever onset
    is_febrile = any(abs(t_days - ft) < (2.0 / 24.0) for ft in fever_times)

    # Temperature
    if is_febrile:
        temp = float(rng.normal(38.8, 0.4))
    else:
        temp = float(rng.normal(36.8, 0.3))
    temp = max(35.0, min(42.0, temp))

    # Heart rate — correlated with fever and neutropenia
    hr_base = 75.0
    if is_febrile:
        hr_base += 20.0
    if anc < 0.5:
        hr_base += 8.0
    hr = float(rng.normal(hr_base, 8.0))
    hr = max(40.0, min(180.0, hr))

    # Respiratory rate — correlated with fever
    rr_base = 16.0 + (4.0 if is_febrile else 0.0)
    rr = float(rng.normal(rr_base, 2.0))
    rr = max(8.0, min(40.0, rr))

    # Blood pressure — mild hypotension with severe infection
    sbp_base = 120.0 - (10.0 if is_febrile and anc < 0.5 else 0.0)
    sbp = float(rng.normal(sbp_base, 10.0))
    sbp = max(70.0, min(200.0, sbp))
    dbp = float(rng.normal(sbp * 0.65, 6.0))
    dbp = max(40.0, min(120.0, dbp))

    # SpO2 — slightly reduced with fever
    spo2 = float(rng.normal(97.5 - (1.5 if is_febrile else 0.0), 1.0))
    spo2 = max(85.0, min(100.0, spo2))

    # GCS — normal unless severe deterioration (rare)
    gcs = 15 if rng.random() > 0.02 else int(rng.integers(10, 15))

    return {
        "temperature_c": round(temp, 1),
        "heart_rate_bpm": round(hr, 0),
        "respiratory_rate_rpm": round(rr, 0),
        "sbp_mmhg": round(sbp, 0),
        "dbp_mmhg": round(dbp, 0),
        "spo2_pct": round(spo2, 1),
        "gcs": gcs,
    }


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _make_vital_record(
    patient_id: str,
    loinc_code: str,
    value: float | int | None,
    unit: str,
    ts: datetime,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system="synthetic",
        source_record_id=str(uuid.uuid4()),
        record_type="vital",
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=VitalRecord(
            loinc_code=loinc_code,
            value_numeric=float(value) if value is not None else None,
            value_text=None,
            unit=unit,
        ),
    )


def _make_lab_record(
    patient_id: str,
    loinc_code: str,
    value: float | None,
    unit: str,
    ts: datetime,
    ref_low: float | None = None,
    ref_high: float | None = None,
    abnormal_flag: str | None = None,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system="synthetic",
        source_record_id=str(uuid.uuid4()),
        record_type="lab",
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=LabRecord(
            loinc_code=loinc_code,
            source_code=loinc_code,
            source_system_code="synthetic",
            value_numeric=value,
            value_text=None,
            unit=unit,
            reference_range_low=ref_low,
            reference_range_high=ref_high,
            abnormal_flag=abnormal_flag,
        ),
    )


def _make_event_record(
    patient_id: str,
    event_type: str,
    ts: datetime,
    metadata: dict[str, Any] | None = None,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system="synthetic",
        source_record_id=str(uuid.uuid4()),
        record_type="event",
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=ClinicalEventRecord(
            event_type=event_type,
            description=None,
            metadata=metadata or {},
        ),
    )


# ---------------------------------------------------------------------------
# Per-patient record generation
# ---------------------------------------------------------------------------

def generate_patient_records(
    patient_id: str,
    regimen: str,
    admission_dt: datetime,
    duration_days: int,
    infection_event_times: list[float],
    rng: np.random.Generator,
    vitals_interval_hours: float = 4.0,
    labs_interval_hours: float = 12.0,
) -> list[CanonicalRecord]:
    """
    Generate all CanonicalRecord objects for a single synthetic patient.

    Returns vitals (every `vitals_interval_hours`), labs (every `labs_interval_hours`),
    and clinical events (admission, infection events).
    """
    params = REGIMEN_PARAMS.get(regimen, REGIMEN_PARAMS["other"])
    records: list[CanonicalRecord] = []

    # Simulate fever times via Hawkes process
    fever_times = simulate_hawkes_fever_times(
        infection_event_times=infection_event_times,
        horizon_days=float(duration_days),
        baseline_intensity=0.02,
        excitation_amplitude=0.8,
        decay_rate=3.0,
        rng=rng,
    )

    # Admission event
    records.append(_make_event_record(patient_id, "admission", admission_dt))

    # Vitals time series
    t = 0.0
    while t <= duration_days:
        ts = admission_dt + timedelta(hours=t * 24.0)
        anc = anc_at_day(t, params, rng)
        vitals = generate_vitals_at_time(t, fever_times, anc, rng)

        records.append(_make_vital_record(patient_id, _LOINC_TEMP, vitals["temperature_c"], "Cel", ts))
        records.append(_make_vital_record(patient_id, _LOINC_HR, vitals["heart_rate_bpm"], "/min", ts))
        records.append(_make_vital_record(patient_id, _LOINC_RR, vitals["respiratory_rate_rpm"], "/min", ts))
        records.append(_make_vital_record(patient_id, _LOINC_SBP, vitals["sbp_mmhg"], "mm[Hg]", ts))
        records.append(_make_vital_record(patient_id, _LOINC_DBP, vitals["dbp_mmhg"], "mm[Hg]", ts))
        records.append(_make_vital_record(patient_id, _LOINC_SPO2, vitals["spo2_pct"], "%", ts))
        records.append(_make_vital_record(patient_id, _LOINC_GCS, vitals["gcs"], "{score}", ts))

        t += vitals_interval_hours / 24.0

    # Labs time series
    t = 0.0
    while t <= duration_days:
        ts = admission_dt + timedelta(hours=t * 24.0)
        anc = anc_at_day(t, params, rng)
        wbc = anc / 0.6 + float(rng.normal(0, 0.3))  # ANC ~60% of WBC
        wbc = max(0.1, wbc)
        crp = float(rng.normal(5.0, 2.0))
        pct = float(rng.normal(0.05, 0.02))

        # Elevate CRP/PCT near infection events
        for t_inf in infection_event_times:
            if 0 <= t - t_inf <= 2.0:
                crp += float(rng.normal(80.0, 20.0))
                pct += float(rng.normal(2.0, 0.5))

        anc_flag = "L" if anc < 1.5 else None
        records.append(_make_lab_record(
            patient_id, _LOINC_ANC, round(anc, 2), "10*9/L",
            ts, ref_low=1.8, ref_high=7.5, abnormal_flag=anc_flag,
        ))
        records.append(_make_lab_record(
            patient_id, _LOINC_WBC, round(max(0.1, wbc), 2), "10*9/L",
            ts, ref_low=4.0, ref_high=11.0,
        ))
        records.append(_make_lab_record(
            patient_id, _LOINC_CRP, round(max(0.1, crp), 1), "mg/L",
            ts, ref_low=None, ref_high=10.0,
        ))
        records.append(_make_lab_record(
            patient_id, _LOINC_PCT, round(max(0.01, pct), 3), "ug/L",
            ts, ref_low=None, ref_high=0.5,
        ))

        t += labs_interval_hours / 24.0

    # Infection events
    for t_inf in infection_event_times:
        ts = admission_dt + timedelta(days=t_inf)
        records.append(_make_event_record(
            patient_id, "infection_confirmed", ts,
            metadata={"source": "synthetic", "day_post_admission": round(t_inf, 2)},
        ))

    return records


# ---------------------------------------------------------------------------
# SyntheticAdapter
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPatientSpec:
    """Specification for a single synthetic patient."""
    patient_id: str
    regimen: str
    admission_dt: datetime
    duration_days: int
    infection_event_times: list[float]  # days from admission
    seed: int | None = None


class SyntheticAdapter:
    """
    SourceAdapter that generates synthetic oncology patient records.

    parse() accepts a JSON-encoded SyntheticPatientSpec (as bytes) and returns
    CanonicalRecord objects in the same format as real ingested data.

    For cohort-level generation, use generate_patient_records() directly.
    """

    source_type: str = "synthetic"

    def parse(self, raw: bytes) -> list[CanonicalRecord]:
        """
        Parse a JSON-encoded SyntheticPatientSpec and generate records.

        Expected JSON keys: patient_id, regimen, admission_dt (ISO 8601),
        duration_days, infection_event_times (list of floats), seed (optional).
        """
        import json
        data = json.loads(raw)
        admission_dt = datetime.fromisoformat(
            data["admission_dt"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        seed = data.get("seed")
        rng = np.random.default_rng(seed)

        return generate_patient_records(
            patient_id=data["patient_id"],
            regimen=data.get("regimen", "other"),
            admission_dt=admission_dt,
            duration_days=int(data.get("duration_days", 30)),
            infection_event_times=list(data.get("infection_event_times", [])),
            rng=rng,
        )

    def validate(self, record: CanonicalRecord) -> ValidationResult:
        errors: list[str] = []
        if not record.patient_id:
            errors.append("patient_id is empty")
        if record.source_system != "synthetic":
            errors.append(f"Expected source_system='synthetic', got '{record.source_system}'")
        if record.timestamp_utc.tzinfo is None:
            errors.append("timestamp_utc must be timezone-aware")
        return ValidationResult(valid=len(errors) == 0, errors=errors)
