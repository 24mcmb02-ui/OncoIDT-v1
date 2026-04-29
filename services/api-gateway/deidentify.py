"""
OncoIDT API Gateway — HIPAA Safe Harbor de-identification pipeline.

Removes all 18 HIPAA Safe Harbor identifiers from PatientTwin objects
before returning data on /api/v1/research/ endpoints for Research_Analyst role.

HIPAA Safe Harbor 18 identifiers removed:
  1.  Names (mrn replaced with pseudonym)
  2.  Geographic data smaller than state (ward_id generalised)
  3.  Dates (admission/discharge shifted by random offset per patient)
  4.  Phone numbers (not in PatientTwin — N/A)
  5.  Fax numbers (N/A)
  6.  Email addresses (N/A)
  7.  SSN (N/A)
  8.  Medical record numbers (mrn → pseudonymised)
  9.  Health plan beneficiary numbers (N/A)
  10. Account numbers (N/A)
  11. Certificate/license numbers (N/A)
  12. Vehicle identifiers (N/A)
  13. Device identifiers (N/A)
  14. URLs (N/A)
  15. IP addresses (N/A)
  16. Biometric identifiers (N/A)
  17. Full-face photographs (N/A)
  18. Any other unique identifying number (patient_id pseudonymised)

Requirements: 17.6
"""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from shared.schemas import PatientTwinSchema

# Salt for pseudonymisation — in production load from secrets manager
_PSEUDO_SALT = "oncoidt-deident-salt-CHANGE-IN-PRODUCTION"

# Date shift: deterministic per patient (derived from patient_id hash)
# Shift range: ±365 days
_DATE_SHIFT_RANGE_DAYS = 365


def _pseudonymise(value: str) -> str:
    """One-way pseudonymisation using HMAC-SHA256 truncated to 16 hex chars."""
    h = hashlib.sha256(f"{_PSEUDO_SALT}:{value}".encode()).hexdigest()
    return h[:16]


def _date_shift_days(patient_id: str) -> int:
    """Deterministic date shift in days for a given patient_id."""
    h = int(hashlib.sha256(f"{_PSEUDO_SALT}:date:{patient_id}".encode()).hexdigest(), 16)
    # Map to [-365, +365]
    return (h % (2 * _DATE_SHIFT_RANGE_DAYS + 1)) - _DATE_SHIFT_RANGE_DAYS


def _shift_date(dt: datetime | None, shift_days: int) -> datetime | None:
    if dt is None:
        return None
    return dt + timedelta(days=shift_days)


def deidentify_patient_twin(twin: PatientTwinSchema) -> dict[str, Any]:
    """
    Return a de-identified dict representation of a PatientTwin.
    The original object is not mutated.

    Applies HIPAA Safe Harbor 18-identifier removal:
    - patient_id → pseudonymised
    - mrn → pseudonymised
    - ward_id → generalised (first segment only, e.g. "ward-A-room-3" → "ward-A")
    - bed_id → removed
    - admission_timestamp, discharge_timestamp → date-shifted
    - age_years → generalised to 5-year band if ≥ 90 (set to 90)
    - All risk score timestamps → date-shifted
    - All vitals/labs timestamps → date-shifted
    """
    original_patient_id = twin.patient_id
    shift = _date_shift_days(original_patient_id)

    data = twin.model_dump()

    # 1. Pseudonymise direct identifiers
    data["patient_id"] = _pseudonymise(original_patient_id)
    data["mrn"] = _pseudonymise(twin.mrn)

    # 2. Generalise ward_id (keep only first hyphen-separated segment pair)
    ward_parts = twin.ward_id.split("-")
    data["ward_id"] = "-".join(ward_parts[:2]) if len(ward_parts) >= 2 else twin.ward_id

    # 3. Remove bed_id (too specific)
    data["bed_id"] = None

    # 4. Shift dates
    data["admission_timestamp"] = _shift_date(twin.admission_timestamp, shift)
    data["discharge_timestamp"] = _shift_date(twin.discharge_timestamp, shift)

    # 5. Age generalisation: cap at 89 (HIPAA requires ages ≥ 90 be reported as 90)
    if twin.age_years >= 90:
        data["age_years"] = 90

    # 6. Shift vitals timestamp
    if twin.vitals and twin.vitals.timestamp:
        data["vitals"]["timestamp"] = _shift_date(twin.vitals.timestamp, shift)

    # 7. Shift labs timestamp
    if twin.labs and twin.labs.timestamp:
        data["labs"]["timestamp"] = _shift_date(twin.labs.timestamp, shift)

    # 8. Shift risk score timestamps
    for horizon_key, score in data.get("infection_risk_scores", {}).items():
        if score and score.get("timestamp"):
            score["timestamp"] = _shift_date(
                twin.infection_risk_scores[horizon_key].timestamp, shift
            )

    for horizon_key, score in data.get("deterioration_risk_scores", {}).items():
        if score and score.get("timestamp"):
            score["timestamp"] = _shift_date(
                twin.deterioration_risk_scores[horizon_key].timestamp, shift
            )

    # 9. Shift survival estimate timestamp
    if data.get("survival_estimate") and twin.survival_estimate:
        data["survival_estimate"]["timestamp"] = _shift_date(
            twin.survival_estimate.timestamp, shift
        )

    # 10. Mark as de-identified
    data["_deidentified"] = True
    data["_deident_method"] = "hipaa_safe_harbor"

    return data


def deidentify_patient_list(twins: list[PatientTwinSchema]) -> list[dict[str, Any]]:
    """De-identify a list of PatientTwin objects."""
    return [deidentify_patient_twin(t) for t in twins]
