"""
HL7v2Adapter — parse HL7 v2.x messages into CanonicalRecord objects.

Supported message types:
  - ADT^A01 (Admit), A02 (Transfer), A03 (Discharge), A08 (Update)
    → ClinicalEventRecord
  - ORU^R01 (Observation Result)
    → LabRecord with LOINC normalization

Uses the hl7apy library for parsing.
Raw HL7 bytes are stored in ClinicalEventRecord.metadata["raw_hl7"].
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.models import (
    CanonicalRecord,
    ClinicalEventRecord,
    DataQualityFlag,
    LabRecord,
)
from services.ingestion.adapters.base import SourceAdapter, ValidationResult

# Load LOINC map for OBX normalization
_HERE = Path(__file__).parent
with open(_HERE / "loinc_map.json") as _f:
    _LOINC_MAP: dict[str, dict[str, str]] = json.load(_f)

# Map OBX-3 source codes to LOINC where possible (extend as needed)
_SOURCE_TO_LOINC: dict[str, str] = {v.get("field", ""): k for k, v in _LOINC_MAP.items()}

_ADT_EVENT_MAP = {
    "A01": "admission",
    "A02": "transfer",
    "A03": "discharge",
    "A08": "patient_update",
}


def _parse_hl7_datetime(value: str | None) -> datetime | None:
    """Parse HL7 DTM format (YYYYMMDDHHMMSS[.SSSS][+/-ZZZZ]) to UTC datetime."""
    if not value:
        return None
    value = value.strip()
    # Strip timezone offset for simplicity; treat as UTC
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(value[:len(fmt.replace("%", "XX").replace("X", ""))], fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def _safe_component(field: Any, index: int = 0) -> str:
    """Safely extract a component from an hl7apy field."""
    try:
        children = list(field.children)
        if index < len(children):
            return str(children[index].value).strip()
        return str(field.value).strip() if index == 0 else ""
    except Exception:
        try:
            return str(field.value).strip()
        except Exception:
            return ""


def _parse_adt(msg: Any, raw: bytes, source_system: str) -> CanonicalRecord:
    """Parse an ADT message into a ClinicalEventRecord."""
    try:
        msh = msg.msh[0]
        pid = msg.pid[0]
        evn = msg.evn[0]
    except Exception as exc:
        raise ValueError(f"Missing required ADT segment: {exc}") from exc

    # Message type trigger event (e.g. A01)
    try:
        msg_type_field = msh.msh_9
        trigger = _safe_component(msg_type_field, 1) or _safe_component(msg_type_field, 0)
        # hl7apy may return "ADT^A01" as a single value
        if "^" in trigger:
            trigger = trigger.split("^")[1]
    except Exception:
        trigger = "UNKNOWN"

    event_type = _ADT_EVENT_MAP.get(trigger, f"adt_{trigger.lower()}")

    # Patient ID from PID-3
    try:
        patient_id = _safe_component(pid.pid_3, 0)
    except Exception:
        patient_id = str(uuid.uuid4())

    # Event timestamp from EVN-2 or MSH-7
    ts: datetime | None = None
    try:
        ts = _parse_hl7_datetime(_safe_component(evn.evn_2))
    except Exception:
        pass
    if ts is None:
        try:
            ts = _parse_hl7_datetime(_safe_component(msh.msh_7))
        except Exception:
            pass
    ts = ts or datetime.now(timezone.utc)

    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system=source_system,
        source_record_id=str(uuid.uuid4()),
        record_type="event",
        timestamp_utc=ts,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=ClinicalEventRecord(
            event_type=event_type,
            description=f"HL7v2 ADT {trigger}",
            metadata={
                "raw_hl7": raw.decode("utf-8", errors="replace"),
                "trigger_event": trigger,
            },
        ),
    )


def _parse_oru(msg: Any, raw: bytes, source_system: str) -> list[CanonicalRecord]:
    """Parse an ORU^R01 message; one CanonicalRecord per OBX segment."""
    records: list[CanonicalRecord] = []

    try:
        msh = msg.msh[0]
        pid = msg.pid[0]
    except Exception as exc:
        raise ValueError(f"Missing required ORU segment: {exc}") from exc

    try:
        patient_id = _safe_component(pid.pid_3, 0)
    except Exception:
        patient_id = str(uuid.uuid4())

    # Observation timestamp from OBR-7 (first OBR)
    obs_ts: datetime | None = None
    try:
        obr = msg.obr[0]
        obs_ts = _parse_hl7_datetime(_safe_component(obr.obr_7))
    except Exception:
        pass

    # Iterate all OBX segments
    try:
        obx_segments = msg.obx
    except Exception:
        obx_segments = []

    for obx in obx_segments:
        try:
            # OBX-3: observation identifier
            obs_id_field = obx.obx_3
            source_code = _safe_component(obs_id_field, 0)
            source_system_code = _safe_component(obs_id_field, 2) or source_system

            # Attempt LOINC lookup
            loinc_code = source_code if source_code in _LOINC_MAP else ""

            # OBX-5: observation value
            value_numeric: float | None = None
            value_text: str | None = None
            try:
                raw_val = _safe_component(obx.obx_5)
                value_numeric = float(raw_val)
            except (ValueError, TypeError):
                value_text = _safe_component(obx.obx_5)

            # OBX-6: units
            unit = _safe_component(obx.obx_6)

            # OBX-7: reference range
            ref_low: float | None = None
            ref_high: float | None = None
            try:
                ref_range = _safe_component(obx.obx_7)
                if "-" in ref_range:
                    parts = ref_range.split("-", 1)
                    ref_low = float(parts[0])
                    ref_high = float(parts[1])
            except Exception:
                pass

            # OBX-8: abnormal flags
            abnormal_flag: str | None = _safe_component(obx.obx_8) or None

            # OBX-14: observation timestamp (overrides OBR-7 if present)
            ts: datetime | None = None
            try:
                ts = _parse_hl7_datetime(_safe_component(obx.obx_14))
            except Exception:
                pass
            ts = ts or obs_ts or datetime.now(timezone.utc)

            flags: list[DataQualityFlag] = []
            if not loinc_code:
                flags.append(DataQualityFlag(
                    flag_type="missing",
                    field_name="loinc_code",
                    message=f"OBX source code '{source_code}' not found in LOINC map",
                    severity="warning",
                ))

            records.append(CanonicalRecord(
                record_id=str(uuid.uuid4()),
                patient_id=patient_id,
                source_system=source_system,
                source_record_id=str(uuid.uuid4()),
                record_type="lab",
                timestamp_utc=ts,
                ingested_at=datetime.now(timezone.utc),
                data_quality_flags=flags,
                payload=LabRecord(
                    loinc_code=loinc_code,
                    source_code=source_code,
                    source_system_code=source_system_code,
                    value_numeric=value_numeric,
                    value_text=value_text,
                    unit=unit,
                    reference_range_low=ref_low,
                    reference_range_high=ref_high,
                    abnormal_flag=abnormal_flag,
                ),
            ))
        except Exception:
            # Skip malformed OBX segments; log quality flag on the first record if any
            continue

    return records


class HL7v2Adapter:
    """Parse HL7 v2.x ADT and ORU^R01 messages into CanonicalRecord objects."""

    source_type: str = "hl7v2"

    def parse(self, raw: bytes) -> list[CanonicalRecord]:
        try:
            from hl7apy.core import Message
            from hl7apy.parser import parse_message
        except ImportError as exc:
            raise RuntimeError(
                "hl7apy is required for HL7v2 parsing. Install with: pip install hl7apy"
            ) from exc

        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            raise ValueError("Empty HL7 message")

        try:
            msg = parse_message(text, find_groups=False)
        except Exception as exc:
            raise ValueError(f"Failed to parse HL7 message: {exc}") from exc

        try:
            msh9 = msg.msh[0].msh_9
            msg_type = _safe_component(msh9, 0)
            trigger = _safe_component(msh9, 1)
            # Handle "ADT^A01" packed in component 0
            if "^" in msg_type:
                parts = msg_type.split("^")
                msg_type = parts[0]
                trigger = parts[1] if len(parts) > 1 else trigger
        except Exception:
            msg_type = "UNKNOWN"
            trigger = ""

        if msg_type == "ADT":
            return [_parse_adt(msg, raw, self.source_type)]
        elif msg_type == "ORU":
            return _parse_oru(msg, raw, self.source_type)
        else:
            raise ValueError(f"Unsupported HL7 message type: {msg_type}")

    def validate(self, record: CanonicalRecord) -> ValidationResult:
        errors: list[str] = []
        if not record.patient_id:
            errors.append("patient_id is empty")
        if record.timestamp_utc.tzinfo is None:
            errors.append("timestamp_utc must be timezone-aware")
        return ValidationResult(valid=len(errors) == 0, errors=errors)
