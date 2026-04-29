"""
JSONPushAdapter — parse JSON push payloads into CanonicalRecord objects.

Accepts:
  - A single CanonicalRecord-shaped JSON object
  - An array of CanonicalRecord-shaped JSON objects

Validates each record against the CanonicalRecordSchema (Pydantic v2).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from shared.models import (
    CanonicalRecord,
    ClinicalEventRecord,
    DataQualityFlag,
    LabRecord,
    MedicationRecord,
    NoteMetadataRecord,
    ObservationRecord,
    VitalRecord,
)
from shared.schemas import CanonicalRecordSchema
from services.ingestion.adapters.base import SourceAdapter, ValidationResult

# Payload type discriminator → dataclass constructor
_PAYLOAD_CONSTRUCTORS = {
    "vital": VitalRecord,
    "lab": LabRecord,
    "medication": MedicationRecord,
    "observation": ObservationRecord,
    "event": ClinicalEventRecord,
    "note_metadata": NoteMetadataRecord,
}


def _schema_to_model(schema: CanonicalRecordSchema) -> CanonicalRecord:
    """Convert a validated Pydantic schema instance to the dataclass model."""
    payload_data = schema.payload.model_dump()
    record_type = schema.record_type
    constructor = _PAYLOAD_CONSTRUCTORS.get(record_type)
    if constructor is None:
        raise ValueError(f"Unknown record_type: {record_type}")
    payload = constructor(**payload_data)

    return CanonicalRecord(
        record_id=schema.record_id,
        patient_id=schema.patient_id,
        source_system=schema.source_system,
        source_record_id=schema.source_record_id,
        record_type=schema.record_type,
        timestamp_utc=schema.timestamp_utc,
        ingested_at=schema.ingested_at,
        data_quality_flags=[
            DataQualityFlag(
                flag_type=f.flag_type,
                field_name=f.field_name,
                message=f.message,
                severity=f.severity,
            )
            for f in schema.data_quality_flags
        ],
        payload=payload,
    )


class JSONPushAdapter:
    """Parse JSON push payloads (single record or array) into CanonicalRecord objects."""

    source_type: str = "json"

    def parse(self, raw: bytes) -> list[CanonicalRecord]:
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        items: list[Any] = data if isinstance(data, list) else [data]
        records: list[CanonicalRecord] = []

        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object, got {type(item).__name__}")

            # Inject defaults for optional fields if absent
            item.setdefault("record_id", str(uuid.uuid4()))
            item.setdefault("source_system", self.source_type)
            item.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())

            try:
                schema = CanonicalRecordSchema.model_validate(item)
            except PydanticValidationError as exc:
                raise ValueError(f"Schema validation failed: {exc}") from exc

            records.append(_schema_to_model(schema))

        return records

    def validate(self, record: CanonicalRecord) -> ValidationResult:
        errors: list[str] = []
        if not record.patient_id:
            errors.append("patient_id is empty")
        if record.timestamp_utc.tzinfo is None:
            errors.append("timestamp_utc must be timezone-aware")
        return ValidationResult(valid=len(errors) == 0, errors=errors)
