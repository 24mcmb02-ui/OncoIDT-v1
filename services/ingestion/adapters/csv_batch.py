"""
CSVBatchAdapter — parse CSV files into CanonicalRecord objects.

Column mapping is driven by a configurable field_mapping dict:
    {
        "patient_id": "PatientID",
        "timestamp_utc": "ObservationTime",
        "record_type": "vital",          # literal or column name
        "loinc_code": "LoincCode",
        "value_numeric": "Value",
        "unit": "Unit",
        ...
    }

The mapping JSON can be passed at construction time or loaded from a file.
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.models import (
    CanonicalRecord,
    DataQualityFlag,
    LabRecord,
    VitalRecord,
)
from services.ingestion.adapters.base import SourceAdapter, ValidationResult

# Default field mapping — maps canonical field names to CSV column names
DEFAULT_FIELD_MAPPING: dict[str, str] = {
    "patient_id": "patient_id",
    "source_record_id": "source_record_id",
    "record_type": "record_type",
    "timestamp_utc": "timestamp_utc",
    "loinc_code": "loinc_code",
    "value_numeric": "value_numeric",
    "value_text": "value_text",
    "unit": "unit",
    "source_code": "source_code",
    "source_system_code": "source_system_code",
    "reference_range_low": "reference_range_low",
    "reference_range_high": "reference_range_high",
    "abnormal_flag": "abnormal_flag",
}


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _get(row: dict[str, str], mapping: dict[str, str], key: str, default: str = "") -> str:
    col = mapping.get(key, key)
    return row.get(col, default).strip()


class CSVBatchAdapter:
    """Parse CSV bytes into CanonicalRecord objects using a configurable field mapping."""

    source_type: str = "csv"

    def __init__(
        self,
        field_mapping: dict[str, str] | None = None,
        mapping_file: str | Path | None = None,
    ) -> None:
        if mapping_file is not None:
            with open(mapping_file) as f:
                self._mapping: dict[str, str] = json.load(f)
        else:
            self._mapping = field_mapping or DEFAULT_FIELD_MAPPING

    def parse(self, raw: bytes) -> list[CanonicalRecord]:
        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        records: list[CanonicalRecord] = []

        for row in reader:
            flags: list[DataQualityFlag] = []

            patient_id = _get(row, self._mapping, "patient_id")
            if not patient_id:
                flags.append(DataQualityFlag(
                    flag_type="missing", field_name="patient_id",
                    message="patient_id is empty", severity="error",
                ))

            ts_str = _get(row, self._mapping, "timestamp_utc")
            ts = _parse_dt(ts_str)
            if ts is None:
                flags.append(DataQualityFlag(
                    flag_type="missing", field_name="timestamp_utc",
                    message=f"Cannot parse timestamp: {ts_str!r}", severity="error",
                ))
                ts = datetime.now(timezone.utc)

            record_type = _get(row, self._mapping, "record_type") or "lab"
            loinc_code = _get(row, self._mapping, "loinc_code")
            source_code = _get(row, self._mapping, "source_code")
            source_system_code = _get(row, self._mapping, "source_system_code") or self.source_type
            unit = _get(row, self._mapping, "unit")

            value_numeric: float | None = None
            raw_val = _get(row, self._mapping, "value_numeric")
            if raw_val:
                try:
                    value_numeric = float(raw_val)
                except ValueError:
                    flags.append(DataQualityFlag(
                        flag_type="out_of_range", field_name="value_numeric",
                        message=f"Cannot parse numeric value: {raw_val!r}", severity="warning",
                    ))

            value_text: str | None = _get(row, self._mapping, "value_text") or None

            ref_low: float | None = None
            ref_high: float | None = None
            try:
                v = _get(row, self._mapping, "reference_range_low")
                ref_low = float(v) if v else None
            except ValueError:
                pass
            try:
                v = _get(row, self._mapping, "reference_range_high")
                ref_high = float(v) if v else None
            except ValueError:
                pass

            abnormal_flag: str | None = _get(row, self._mapping, "abnormal_flag") or None

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
                    source_code=source_code,
                    source_system_code=source_system_code,
                    value_numeric=value_numeric,
                    value_text=value_text,
                    unit=unit,
                    reference_range_low=ref_low,
                    reference_range_high=ref_high,
                    abnormal_flag=abnormal_flag,
                )

            records.append(CanonicalRecord(
                record_id=str(uuid.uuid4()),
                patient_id=patient_id,
                source_system=self.source_type,
                source_record_id=_get(row, self._mapping, "source_record_id") or str(uuid.uuid4()),
                record_type=record_type,  # type: ignore[arg-type]
                timestamp_utc=ts,
                ingested_at=datetime.now(timezone.utc),
                data_quality_flags=flags,
                payload=payload,
            ))

        return records

    def validate(self, record: CanonicalRecord) -> ValidationResult:
        errors: list[str] = []
        if not record.patient_id:
            errors.append("patient_id is empty")
        if record.timestamp_utc.tzinfo is None:
            errors.append("timestamp_utc must be timezone-aware")
        return ValidationResult(valid=len(errors) == 0, errors=errors)
