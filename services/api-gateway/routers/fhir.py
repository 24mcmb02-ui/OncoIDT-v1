"""
OncoIDT API Gateway — FHIR R4 endpoints.

Endpoints:
  GET /api/v1/fhir/Patient/{id}              — PatientTwin → FHIR R4 Patient
  GET /api/v1/fhir/Observation               — clinical events → FHIR R4 Observation Bundle
  GET /api/v1/fhir/RiskAssessment/{id}       — RiskScore → FHIR R4 RiskAssessment
  GET /api/v1/fhir/MedicationAdministration  — medication records → FHIR R4 Bundle

All responses use Content-Type: application/fhir+json.

Requirements: 13.6
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from shared.auth import CurrentUser
from shared.config import get_settings
from shared.db import DbSession

logger = logging.getLogger(__name__)
settings = get_settings()

FHIR_CONTENT_TYPE = "application/fhir+json"

router = APIRouter(prefix="/api/v1/fhir", tags=["fhir"])


# ---------------------------------------------------------------------------
# FHIR serialisation helpers
# ---------------------------------------------------------------------------

def _fhir_patient(twin: dict[str, Any]) -> dict[str, Any]:
    """Serialize a PatientTwin dict to a FHIR R4 Patient resource."""
    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": twin["patient_id"],
        "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]},
        "identifier": [
            {
                "system": "urn:oncoidt:patient",
                "value": twin["patient_id"],
            }
        ],
        "gender": _fhir_gender(twin.get("sex", "U")),
        "extension": [
            {
                "url": "urn:oncoidt:chemo-regimen",
                "valueString": twin.get("chemo_regimen", ""),
            },
            {
                "url": "urn:oncoidt:chemo-cycle-phase",
                "valueString": twin.get("chemo_cycle_phase", ""),
            },
            {
                "url": "urn:oncoidt:immunosuppression-score",
                "valueDecimal": twin.get("immunosuppression_score", 0.0),
            },
        ],
    }

    # Admission date
    if twin.get("admission_timestamp"):
        resource["extension"].append({
            "url": "urn:oncoidt:admission-timestamp",
            "valueDateTime": _iso(twin["admission_timestamp"]),
        })

    # Primary diagnosis
    if twin.get("primary_diagnosis_icd10"):
        resource["extension"].append({
            "url": "urn:oncoidt:primary-diagnosis-icd10",
            "valueCode": twin["primary_diagnosis_icd10"],
        })

    return resource


def _fhir_observation(record: dict[str, Any], patient_id: str) -> dict[str, Any]:
    """Serialize a clinical event row to a FHIR R4 Observation resource."""
    payload = record.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)

    obs: dict[str, Any] = {
        "resourceType": "Observation",
        "id": record.get("record_id", ""),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": _iso(record.get("timestamp_utc")),
    }

    loinc = payload.get("loinc_code")
    if loinc:
        obs["code"] = {
            "coding": [{"system": "http://loinc.org", "code": loinc}]
        }

    value = payload.get("value_numeric")
    unit = payload.get("unit", "")
    if value is not None:
        obs["valueQuantity"] = {"value": value, "unit": unit}
    elif payload.get("value_text"):
        obs["valueString"] = payload["value_text"]

    return obs


def _fhir_risk_assessment(
    score_row: dict[str, Any],
    patient_id: str,
) -> dict[str, Any]:
    """Serialize a risk_scores row to a FHIR R4 RiskAssessment resource."""
    return {
        "resourceType": "RiskAssessment",
        "id": score_row.get("score_id", ""),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "occurrenceDateTime": _iso(score_row.get("timestamp")),
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "urn:oncoidt:score-type",
                            "code": score_row.get("score_type", ""),
                        }
                    ]
                },
                "probabilityDecimal": score_row.get("score", 0.0),
                "whenRange": {
                    "high": {
                        "value": score_row.get("forecast_horizon_hours", 24),
                        "unit": "h",
                        "system": "http://unitsofmeasure.org",
                        "code": "h",
                    }
                },
            }
        ],
        "extension": [
            {
                "url": "urn:oncoidt:uncertainty-lower",
                "valueDecimal": score_row.get("uncertainty_lower", 0.0),
            },
            {
                "url": "urn:oncoidt:uncertainty-upper",
                "valueDecimal": score_row.get("uncertainty_upper", 1.0),
            },
            {
                "url": "urn:oncoidt:model-version",
                "valueString": score_row.get("model_version", ""),
            },
        ],
    }


def _fhir_medication_administration(record: dict[str, Any], patient_id: str) -> dict[str, Any]:
    """Serialize a medication record to a FHIR R4 MedicationAdministration resource."""
    payload = record.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)

    resource: dict[str, Any] = {
        "resourceType": "MedicationAdministration",
        "id": record.get("record_id", ""),
        "status": "completed",
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": _iso(payload.get("administration_timestamp") or record.get("timestamp_utc")),
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": payload.get("rxnorm_code", ""),
                    "display": payload.get("drug_name", ""),
                }
            ]
        },
    }

    if payload.get("dose_mg") is not None:
        resource["dosage"] = {
            "route": {"text": payload.get("route", "")},
            "dose": {
                "value": payload["dose_mg"],
                "unit": "mg",
                "system": "http://unitsofmeasure.org",
                "code": "mg",
            },
        }

    if payload.get("is_chemotherapy"):
        resource["extension"] = [
            {"url": "urn:oncoidt:is-chemotherapy", "valueBoolean": True},
            {"url": "urn:oncoidt:chemo-regimen-code", "valueString": payload.get("chemo_regimen_code", "")},
        ]

    return resource


def _fhir_bundle(resources: list[dict[str, Any]], bundle_type: str = "searchset") -> dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": bundle_type,
        "total": len(resources),
        "entry": [{"resource": r} for r in resources],
    }


def _iso(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _fhir_gender(sex: str) -> str:
    return {"M": "male", "F": "female", "O": "other", "U": "unknown"}.get(sex, "unknown")


def _fhir_response(data: dict[str, Any]) -> Response:
    return Response(
        content=json.dumps(data, default=str),
        media_type=FHIR_CONTENT_TYPE,
    )


# ---------------------------------------------------------------------------
# Upstream fetch helpers
# ---------------------------------------------------------------------------

async def _fetch_patient(patient_id: str, current_user: Any) -> dict[str, Any]:
    """Fetch PatientTwin from inference service (or DB directly)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{settings.inference_service_url}/inference/scores/{patient_id}",
            timeout=5.0,
        )
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Upstream service error")
    return resp.json()


# ---------------------------------------------------------------------------
# GET /api/v1/fhir/Patient/{id}
# ---------------------------------------------------------------------------

@router.get("/Patient/{patient_id}", summary="FHIR R4 Patient resource")
async def get_fhir_patient(
    patient_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> Response:
    """Return a PatientTwin serialized as a FHIR R4 Patient resource."""
    from sqlalchemy import text

    result = await session.execute(
        text(
            """
            SELECT p.patient_id, p.mrn, p.ward_id, p.bed_id,
                   p.admission_timestamp, p.discharge_timestamp, p.status,
                   p.age_years, p.sex, p.primary_diagnosis_icd10,
                   p.chemo_regimen, p.chemo_cycle_phase, p.immunosuppression_score
            FROM patient_twins p
            WHERE p.patient_id = :patient_id
            """
        ),
        {"patient_id": patient_id},
    )
    row = result.mappings().fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    if not current_user.can_access_ward(row["ward_id"]):
        raise HTTPException(status_code=403, detail="Access denied to this patient's ward")

    return _fhir_response(_fhir_patient(dict(row)))


# ---------------------------------------------------------------------------
# GET /api/v1/fhir/Observation
# ---------------------------------------------------------------------------

@router.get("/Observation", summary="FHIR R4 Observation Bundle")
async def get_fhir_observations(
    session: DbSession,
    current_user: CurrentUser,
    patient_id: str = Query(..., description="Patient ID to retrieve observations for"),
    record_type: str = Query(default="vital", description="Record type: vital or lab"),
    limit: int = Query(default=100, ge=1, le=500),
) -> Response:
    """Return clinical events for a patient as a FHIR R4 Observation Bundle."""
    from sqlalchemy import text

    # Verify ward access
    ward_result = await session.execute(
        text("SELECT ward_id FROM patient_twins WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    ward_row = ward_result.fetchone()
    if ward_row and not current_user.can_access_ward(ward_row[0]):
        raise HTTPException(status_code=403, detail="Access denied to this patient's ward")

    result = await session.execute(
        text(
            """
            SELECT record_id, patient_id, record_type, timestamp_utc, payload
            FROM clinical_events
            WHERE patient_id = :patient_id AND record_type = :record_type
            ORDER BY timestamp_utc DESC
            LIMIT :limit
            """
        ),
        {"patient_id": patient_id, "record_type": record_type, "limit": limit},
    )
    rows = result.mappings().fetchall()

    observations = [_fhir_observation(dict(r), patient_id) for r in rows]
    return _fhir_response(_fhir_bundle(observations))


# ---------------------------------------------------------------------------
# GET /api/v1/fhir/RiskAssessment/{id}
# ---------------------------------------------------------------------------

@router.get("/RiskAssessment/{score_id}", summary="FHIR R4 RiskAssessment resource")
async def get_fhir_risk_assessment(
    score_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> Response:
    """Return a RiskScore serialized as a FHIR R4 RiskAssessment resource."""
    from sqlalchemy import text

    result = await session.execute(
        text(
            """
            SELECT rs.score_id, rs.patient_id, rs.score_type, rs.forecast_horizon_hours,
                   rs.score, rs.uncertainty_lower, rs.uncertainty_upper,
                   rs.model_version, rs.timestamp,
                   pt.ward_id
            FROM risk_scores rs
            JOIN patient_twins pt ON pt.patient_id = rs.patient_id
            WHERE rs.score_id = :score_id
            """
        ),
        {"score_id": score_id},
    )
    row = result.mappings().fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"RiskAssessment {score_id} not found")

    if not current_user.can_access_ward(row["ward_id"]):
        raise HTTPException(status_code=403, detail="Access denied to this patient's ward")

    return _fhir_response(_fhir_risk_assessment(dict(row), row["patient_id"]))


# ---------------------------------------------------------------------------
# GET /api/v1/fhir/MedicationAdministration
# ---------------------------------------------------------------------------

@router.get("/MedicationAdministration", summary="FHIR R4 MedicationAdministration Bundle")
async def get_fhir_medication_administrations(
    session: DbSession,
    current_user: CurrentUser,
    patient_id: str = Query(..., description="Patient ID"),
    limit: int = Query(default=100, ge=1, le=500),
) -> Response:
    """Return medication records for a patient as a FHIR R4 Bundle."""
    from sqlalchemy import text

    ward_result = await session.execute(
        text("SELECT ward_id FROM patient_twins WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    ward_row = ward_result.fetchone()
    if ward_row and not current_user.can_access_ward(ward_row[0]):
        raise HTTPException(status_code=403, detail="Access denied to this patient's ward")

    result = await session.execute(
        text(
            """
            SELECT record_id, patient_id, timestamp_utc, payload
            FROM clinical_events
            WHERE patient_id = :patient_id AND record_type = 'medication'
            ORDER BY timestamp_utc DESC
            LIMIT :limit
            """
        ),
        {"patient_id": patient_id, "limit": limit},
    )
    rows = result.mappings().fetchall()

    resources = [_fhir_medication_administration(dict(r), patient_id) for r in rows]
    return _fhir_response(_fhir_bundle(resources))
