"""
FHIRAdapter — parse FHIR R4 resources into CanonicalRecord objects.

Supports:
  - Bundle (collection of resources)
  - Patient
  - Observation  (mapped to VitalRecord or LabRecord via LOINC)
  - MedicationAdministration (mapped to MedicationRecord via RxNorm)

Also provides serialize_to_fhir() for round-trip export.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from fastapi import HTTPException

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
# Load bundled lookup tables
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

with open(_HERE / "loinc_map.json") as _f:
    _LOINC_MAP: dict[str, dict[str, str]] = json.load(_f)

with open(_HERE / "chemo_rxnorm.json") as _f:
    _CHEMO_RXNORM: set[str] = set(json.load(_f))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _require_dt(value: str | None, field: str) -> datetime:
    dt = _parse_dt(value)
    if dt is None:
        raise HTTPException(
            status_code=422,
            detail=f"Missing or invalid datetime for field '{field}': {value!r}",
        )
    return dt


def _get_loinc(coding_list: list[dict[str, Any]]) -> str | None:
    for coding in coding_list:
        if coding.get("system", "").endswith("loinc.org"):
            return coding.get("code")
    return None


def _get_rxnorm(coding_list: list[dict[str, Any]]) -> str | None:
    for coding in coding_list:
        if "rxnorm" in coding.get("system", "").lower():
            return coding.get("code")
    return None


def _source_code(coding_list: list[dict[str, Any]]) -> str:
    if coding_list:
        return coding_list[0].get("code", "")
    return ""


# ---------------------------------------------------------------------------
# Resource parsers
# ---------------------------------------------------------------------------

def _parse_observation(resource: dict[str, Any], source_system: str) -> CanonicalRecord:
    """Map a FHIR R4 Observation to a VitalRecord or LabRecord."""
    res_id = resource.get("id") or str(uuid.uuid4())
    patient_ref = (resource.get("subject") or {}).get("reference", "")
    patient_id = patient_ref.split("/")[-1] if patient_ref else ""

    effective = resource.get("effectiveDateTime") or resource.get("effectivePeriod", {}).get("start")
    ts = _require_dt(effective, "effectiveDateTime")

    coding_list: list[dict[str, Any]] = (resource.get("code") or {}).get("coding", [])
    loinc_code = _get_loinc(coding_list) or ""
    src_code = _source_code(coding_list)

    value_numeric: float | None = None
    value_text: str | None = None
    unit = ""
    ref_low: float | None = None
    ref_high: float | None = None
    abnormal_flag: str | None = None

    if "valueQuantity" in resource:
        vq = resource["valueQuantity"]
        value_numeric = vq.get("value")
        unit = vq.get("unit") or vq.get("code") or ""
    elif "valueString" in resource:
        value_text = resource["valueString"]
    elif "valueCodeableConcept" in resource:
        value_text = (resource["valueCodeableConcept"].get("text") or
                      _source_code(resource["valueCodeableConcept"].get("coding", [])))

    for rr in resource.get("referenceRange", []):
        ref_low = (rr.get("low") or {}).get("value")
        ref_high = (rr.get("high") or {}).get("value")
        break

    interp = resource.get("interpretation", [])
    if interp:
        abnormal_flag = _source_code((interp[0] or {}).get("coding", []))

    loinc_info = _LOINC_MAP.get(loinc_code, {})
    record_type: str = loinc_info.get("type", "lab")

    flags: list[DataQualityFlag] = []
    if not loinc_code:
        flags.append(DataQualityFlag(
            flag_type="missing",
            field_name="loinc_code",
            message="No LOINC code found in Observation.code.coding",
            severity="warning",
        ))

    if record_type == "vital":
        payload: VitalRecord | LabRecord = VitalRecord(
            loinc_code=loinc_code,
            value_numeric=value_numeric,
            value_text=value_text,
            unit=unit,
        )
    else:
        payload = LabRecord(
            loinc_code=loinc_code,
            source_code=src_code,
            source_system_code=source_system,
            value_numeric=value_numeric,
            value_text=value_text,
            unit=unit,
            reference_range_low=ref_low,
            reference_range_high=ref_high,
            abnormal_flag=abnormal_flag,
        )

    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system=source_system,
        source_record_id=res_id,
        record_type=record_type,  # type: ignore[arg-type]
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=flags,
        payload=payload,
    )


def _parse_medication_administration(
    resource: dict[str, Any], source_system: str
) -> CanonicalRecord:
    """Map a FHIR R4 MedicationAdministration to a MedicationRecord."""
    res_id = resource.get("id") or str(uuid.uuid4())
    patient_ref = (resource.get("subject") or {}).get("reference", "")
    patient_id = patient_ref.split("/")[-1] if patient_ref else ""

    effective = resource.get("effectiveDateTime") or (
        resource.get("effectivePeriod") or {}
    ).get("start")
    ts = _require_dt(effective, "effectiveDateTime")

    med_ref = resource.get("medicationCodeableConcept") or {}
    coding_list: list[dict[str, Any]] = med_ref.get("coding", [])
    rxnorm_code = _get_rxnorm(coding_list) or ""
    src_code = _source_code(coding_list)
    drug_name = med_ref.get("text") or (coding_list[0].get("display") if coding_list else "") or ""

    dose_mg: float | None = None
    route = ""
    dosage = resource.get("dosage") or {}
    if "dose" in dosage:
        dose_mg = dosage["dose"].get("value")
    if "route" in dosage:
        route_codings = dosage["route"].get("coding", [])
        route = (dosage["route"].get("text") or
                 (route_codings[0].get("display") if route_codings else ""))

    is_chemo = rxnorm_code in _CHEMO_RXNORM
    chemo_regimen_code: str | None = None
    if is_chemo:
        chemo_regimen_code = rxnorm_code

    flags: list[DataQualityFlag] = []
    if not rxnorm_code:
        flags.append(DataQualityFlag(
            flag_type="missing",
            field_name="rxnorm_code",
            message="No RxNorm code found in MedicationAdministration.medicationCodeableConcept",
            severity="warning",
        ))

    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system=source_system,
        source_record_id=res_id,
        record_type="medication",
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=flags,
        payload=MedicationRecord(
            rxnorm_code=rxnorm_code,
            source_code=src_code,
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            is_chemotherapy=is_chemo,
            chemo_regimen_code=chemo_regimen_code,
            administration_timestamp=ts,
        ),
    )


def _parse_patient_resource(resource: dict[str, Any], source_system: str) -> CanonicalRecord:
    """Map a FHIR R4 Patient to a ClinicalEventRecord (admission placeholder)."""
    res_id = resource.get("id") or str(uuid.uuid4())
    ts_str = resource.get("meta", {}).get("lastUpdated")
    ts = _parse_dt(ts_str) or datetime.now(timezone.utc)

    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=res_id,
        source_system=source_system,
        source_record_id=res_id,
        record_type="event",
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=ClinicalEventRecord(
            event_type="patient_registered",
            description=None,
            metadata={"fhir_resource": "Patient", "fhir_id": res_id},
        ),
    )


# ---------------------------------------------------------------------------
# Serialization helpers (round-trip export)
# ---------------------------------------------------------------------------

def _vital_to_fhir(record: CanonicalRecord) -> dict[str, Any]:
    p = record.payload
    assert isinstance(p, VitalRecord)
    return {
        "resourceType": "Observation",
        "id": record.source_record_id,
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": p.loinc_code}]},
        "subject": {"reference": f"Patient/{record.patient_id}"},
        "effectiveDateTime": record.timestamp_utc.isoformat(),
        "valueQuantity": {"value": p.value_numeric, "unit": p.unit},
    }


def _lab_to_fhir(record: CanonicalRecord) -> dict[str, Any]:
    p = record.payload
    assert isinstance(p, LabRecord)
    out: dict[str, Any] = {
        "resourceType": "Observation",
        "id": record.source_record_id,
        "status": "final",
        "code": {
            "coding": [
                {"system": "http://loinc.org", "code": p.loinc_code},
                {"system": p.source_system_code, "code": p.source_code},
            ]
        },
        "subject": {"reference": f"Patient/{record.patient_id}"},
        "effectiveDateTime": record.timestamp_utc.isoformat(),
    }
    if p.value_numeric is not None:
        out["valueQuantity"] = {"value": p.value_numeric, "unit": p.unit}
    elif p.value_text is not None:
        out["valueString"] = p.value_text
    if p.reference_range_low is not None or p.reference_range_high is not None:
        rr: dict[str, Any] = {}
        if p.reference_range_low is not None:
            rr["low"] = {"value": p.reference_range_low}
        if p.reference_range_high is not None:
            rr["high"] = {"value": p.reference_range_high}
        out["referenceRange"] = [rr]
    if p.abnormal_flag:
        out["interpretation"] = [{"coding": [{"code": p.abnormal_flag}]}]
    return out


def _medication_to_fhir(record: CanonicalRecord) -> dict[str, Any]:
    p = record.payload
    assert isinstance(p, MedicationRecord)
    out: dict[str, Any] = {
        "resourceType": "MedicationAdministration",
        "id": record.source_record_id,
        "status": "completed",
        "medicationCodeableConcept": {
            "coding": [
                {"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": p.rxnorm_code},
                {"system": "source", "code": p.source_code},
            ],
            "text": p.drug_name,
        },
        "subject": {"reference": f"Patient/{record.patient_id}"},
        "effectiveDateTime": p.administration_timestamp.isoformat(),
        "dosage": {
            "dose": {"value": p.dose_mg, "unit": "mg"},
            "route": {"text": p.route},
        },
    }
    return out


def serialize_to_fhir(record: CanonicalRecord) -> dict[str, Any]:
    """Serialize a CanonicalRecord back to a FHIR R4 resource dict."""
    if record.record_type == "vital":
        return _vital_to_fhir(record)
    if record.record_type == "lab":
        return _lab_to_fhir(record)
    if record.record_type == "medication":
        return _medication_to_fhir(record)
    raise ValueError(f"Cannot serialize record_type '{record.record_type}' to FHIR")


# ---------------------------------------------------------------------------
# FHIRAdapter
# ---------------------------------------------------------------------------

class FHIRAdapter:
    """Parse FHIR R4 Bundle / individual resources into CanonicalRecord objects."""

    source_type: str = "fhir"

    def parse(self, raw: bytes) -> list[CanonicalRecord]:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc

        resource_type = data.get("resourceType")
        if resource_type is None:
            raise HTTPException(status_code=422, detail="Missing 'resourceType' field")

        resources: list[dict[str, Any]] = []
        if resource_type == "Bundle":
            for entry in data.get("entry", []):
                res = entry.get("resource")
                if res:
                    resources.append(res)
        else:
            resources = [data]

        records: list[CanonicalRecord] = []
        for res in resources:
            rt = res.get("resourceType")
            try:
                if rt == "Observation":
                    records.append(_parse_observation(res, self.source_type))
                elif rt == "MedicationAdministration":
                    records.append(_parse_medication_administration(res, self.source_type))
                elif rt == "Patient":
                    records.append(_parse_patient_resource(res, self.source_type))
                # Other resource types silently skipped
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to parse {rt} resource: {exc}",
                ) from exc

        return records

    def validate(self, record: CanonicalRecord) -> ValidationResult:
        errors: list[str] = []
        if not record.patient_id:
            errors.append("patient_id is empty")
        if record.timestamp_utc.tzinfo is None:
            errors.append("timestamp_utc must be timezone-aware")
        if record.record_type not in ("vital", "lab", "medication", "observation", "event", "note_metadata"):
            errors.append(f"Unknown record_type: {record.record_type}")
        return ValidationResult(valid=len(errors) == 0, errors=errors)
