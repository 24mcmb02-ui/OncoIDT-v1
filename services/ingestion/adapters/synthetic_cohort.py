"""
Cohort configuration loader and ground truth label writer for synthetic data generation.

Loads a YAML config matching the `synthetic_cohort` schema from the design doc,
generates a full patient cohort, and writes ground_truth_labels.csv.
"""
from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic import model_validator
except ImportError:  # pragma: no cover
    raise ImportError("pydantic is required for synthetic_cohort loader")

from shared.models import CanonicalRecord
from services.ingestion.adapters.synthetic import (
    REGIMEN_PARAMS,
    SyntheticAdapter,
    generate_patient_records,
)


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------

class WardLayout(BaseModel):
    n_rooms: int = Field(default=10, ge=1)
    beds_per_room: int = Field(default=4, ge=1)


class EventRates(BaseModel):
    infection_per_admission: float = Field(default=0.12, ge=0.0, le=1.0)
    neutropenic_fever_per_admission: float = Field(default=0.18, ge=0.0, le=1.0)
    deterioration_per_admission: float = Field(default=0.08, ge=0.0, le=1.0)


class ComorbidityPrevalence(BaseModel):
    diabetes: float = Field(default=0.15, ge=0.0, le=1.0)
    hypertension: float = Field(default=0.30, ge=0.0, le=1.0)
    ckd: float = Field(default=0.10, ge=0.0, le=1.0)


class SyntheticCohortConfig(BaseModel):
    n_patients: int = Field(default=200, ge=1)
    ward_layout: WardLayout = Field(default_factory=WardLayout)
    chemo_regimen_distribution: dict[str, float] = Field(
        default_factory=lambda: {"R-CHOP": 0.4, "BEP": 0.2, "FOLFOX": 0.2, "other": 0.2}
    )
    comorbidity_prevalence: ComorbidityPrevalence = Field(
        default_factory=ComorbidityPrevalence
    )
    event_rates: EventRates = Field(default_factory=EventRates)
    simulation_duration_days: int = Field(default=90, ge=1)
    seed: int | None = Field(default=None)

    @field_validator("chemo_regimen_distribution")
    @classmethod
    def _validate_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"chemo_regimen_distribution must sum to 1.0, got {total:.3f}"
            )
        return v


# ---------------------------------------------------------------------------
# Ground truth label row
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthLabel:
    patient_id: str
    label_type: str          # "infection_event" | "neutropenic_fever" | "deterioration_event"
    event_timestamp: datetime
    horizon_6h: bool
    horizon_12h: bool
    horizon_24h: bool
    horizon_48h: bool


# ---------------------------------------------------------------------------
# Cohort generator
# ---------------------------------------------------------------------------

@dataclass
class GeneratedCohort:
    records: list[CanonicalRecord]
    labels: list[GroundTruthLabel]
    patient_ids: list[str]


def _sample_regimen(
    distribution: dict[str, float], rng: np.random.Generator
) -> str:
    regimens = list(distribution.keys())
    probs = [distribution[r] for r in regimens]
    idx = int(rng.choice(len(regimens), p=probs))
    return regimens[idx]


def _sample_infection_times(
    duration_days: int,
    infection_rate: float,
    rng: np.random.Generator,
) -> list[float]:
    """Sample 0 or more infection event times (days from admission)."""
    n_events = int(rng.binomial(1, infection_rate))
    if n_events == 0:
        return []
    # Infection events tend to cluster around nadir (days 7–14)
    times = []
    for _ in range(n_events):
        t = float(rng.normal(10.0, 3.0))
        t = max(1.0, min(float(duration_days - 1), t))
        times.append(round(t, 2))
    return sorted(times)


def _sample_deterioration_times(
    duration_days: int,
    deterioration_rate: float,
    infection_times: list[float],
    rng: np.random.Generator,
) -> list[float]:
    """Deterioration events may follow infection events or occur independently."""
    n_events = int(rng.binomial(1, deterioration_rate))
    if n_events == 0:
        return []
    times = []
    for _ in range(n_events):
        if infection_times and rng.random() < 0.6:
            # Deterioration follows infection by 6–48h
            base = float(rng.choice(infection_times))
            t = base + float(rng.uniform(0.25, 2.0))
        else:
            t = float(rng.uniform(2.0, float(duration_days - 1)))
        t = max(0.5, min(float(duration_days - 0.1), t))
        times.append(round(t, 2))
    return sorted(times)


def _make_labels(
    patient_id: str,
    admission_dt: datetime,
    infection_times: list[float],
    deterioration_times: list[float],
    neutropenic_fever_times: list[float],
) -> list[GroundTruthLabel]:
    labels: list[GroundTruthLabel] = []

    def _horizons(event_t: float, ref_t: float) -> tuple[bool, bool, bool, bool]:
        delta_h = (event_t - ref_t) * 24.0
        return (
            delta_h <= 6.0,
            delta_h <= 12.0,
            delta_h <= 24.0,
            delta_h <= 48.0,
        )

    for t_inf in infection_times:
        event_ts = admission_dt + timedelta(days=t_inf)
        h6, h12, h24, h48 = _horizons(t_inf, t_inf)
        labels.append(GroundTruthLabel(
            patient_id=patient_id,
            label_type="infection_event",
            event_timestamp=event_ts,
            horizon_6h=h6,
            horizon_12h=h12,
            horizon_24h=h24,
            horizon_48h=h48,
        ))

    for t_det in deterioration_times:
        event_ts = admission_dt + timedelta(days=t_det)
        h6, h12, h24, h48 = _horizons(t_det, t_det)
        labels.append(GroundTruthLabel(
            patient_id=patient_id,
            label_type="deterioration_event",
            event_timestamp=event_ts,
            horizon_6h=h6,
            horizon_12h=h12,
            horizon_24h=h24,
            horizon_48h=h48,
        ))

    for t_nf in neutropenic_fever_times:
        event_ts = admission_dt + timedelta(days=t_nf)
        h6, h12, h24, h48 = _horizons(t_nf, t_nf)
        labels.append(GroundTruthLabel(
            patient_id=patient_id,
            label_type="neutropenic_fever",
            event_timestamp=event_ts,
            horizon_6h=h6,
            horizon_12h=h12,
            horizon_24h=h24,
            horizon_48h=h48,
        ))

    return labels


def generate_cohort(config: SyntheticCohortConfig) -> GeneratedCohort:
    """
    Generate a full synthetic patient cohort from config.

    Returns all CanonicalRecord objects and ground truth labels.
    """
    rng = np.random.default_rng(config.seed)
    base_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    all_records: list[CanonicalRecord] = []
    all_labels: list[GroundTruthLabel] = []
    patient_ids: list[str] = []

    for i in range(config.n_patients):
        patient_id = str(uuid.uuid4())
        patient_ids.append(patient_id)

        regimen = _sample_regimen(config.chemo_regimen_distribution, rng)
        # Stagger admissions across the simulation window
        admission_offset_days = float(rng.uniform(0, config.simulation_duration_days * 0.7))
        admission_dt = base_dt + timedelta(days=admission_offset_days)
        duration_days = int(rng.integers(14, min(60, config.simulation_duration_days)))

        infection_times = _sample_infection_times(
            duration_days, config.event_rates.infection_per_admission, rng
        )
        deterioration_times = _sample_deterioration_times(
            duration_days, config.event_rates.deterioration_per_admission,
            infection_times, rng,
        )
        # Neutropenic fever: subset of infection times + independent events
        nf_rate = config.event_rates.neutropenic_fever_per_admission
        nf_times = _sample_infection_times(duration_days, nf_rate, rng)

        records = generate_patient_records(
            patient_id=patient_id,
            regimen=regimen,
            admission_dt=admission_dt,
            duration_days=duration_days,
            infection_event_times=infection_times,
            rng=rng,
        )
        labels = _make_labels(
            patient_id=patient_id,
            admission_dt=admission_dt,
            infection_times=infection_times,
            deterioration_times=deterioration_times,
            neutropenic_fever_times=nf_times,
        )

        all_records.extend(records)
        all_labels.extend(labels)

    return GeneratedCohort(
        records=all_records,
        labels=all_labels,
        patient_ids=patient_ids,
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_ground_truth_labels(labels: list[GroundTruthLabel], output_path: Path) -> None:
    """Write ground truth labels to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "label_type", "event_timestamp",
            "horizon_6h", "horizon_12h", "horizon_24h", "horizon_48h",
        ])
        writer.writeheader()
        for label in labels:
            writer.writerow({
                "patient_id": label.patient_id,
                "label_type": label.label_type,
                "event_timestamp": label.event_timestamp.isoformat(),
                "horizon_6h": label.horizon_6h,
                "horizon_12h": label.horizon_12h,
                "horizon_24h": label.horizon_24h,
                "horizon_48h": label.horizon_48h,
            })


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_cohort_config(path: Path) -> SyntheticCohortConfig:
    """Load a YAML cohort config file and return a validated SyntheticCohortConfig."""
    if yaml is None:
        raise ImportError("PyYAML is required to load cohort config files")
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Support top-level 'synthetic_cohort' key or bare config
    data: dict[str, Any] = raw.get("synthetic_cohort", raw)
    return SyntheticCohortConfig.model_validate(data)
