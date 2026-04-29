"""
Property test — HL7v2 round-trip fidelity.

Property 3 (Requirement 21.5):
    Parsing the same ORU^R01 message twice produces identical CanonicalRecord
    objects (comparing only deterministic/stable fields).

Fields excluded from comparison (intentionally non-deterministic):
    record_id, source_record_id, ingested_at

**Validates: Requirements 21.5**
"""
from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings as h_settings
from hypothesis import strategies as st

from services.ingestion.adapters.hl7v2 import HL7v2Adapter
from shared.models import CanonicalRecord, LabRecord

# ---------------------------------------------------------------------------
# Hypothesis strategies for ORU^R01 messages
# ---------------------------------------------------------------------------

_SOURCE_CODES = ["26499-4", "6690-2", "8310-5", "731-0"]
_UNITS = ["10*9/L", "Cel", "mg/L", "/min", "%"]

_patient_id_strategy = st.from_regex(r"patient-[a-z0-9]{6}", fullmatch=True)
_alphanum_strategy = st.from_regex(r"[A-Z0-9]{6}", fullmatch=True)


@st.composite
def _timestamp_strategy(draw: Any) -> str:
    """Generate a valid HL7 DTM timestamp (YYYYMMDDHHMMSS)."""
    year = draw(st.integers(min_value=2020, max_value=2025))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))
    return f"{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}"


@st.composite
def _obx_segment_strategy(draw: Any, seq: int) -> str:
    """Generate a single OBX segment line."""
    source_code = draw(st.sampled_from(_SOURCE_CODES))
    value = draw(st.floats(min_value=0.1, max_value=999.0, allow_nan=False, allow_infinity=False))
    unit = draw(st.sampled_from(_UNITS))
    # Round to 4 decimal places to avoid float repr issues
    value_str = f"{value:.4f}"
    return f"OBX|{seq}|NM|{source_code}^OBS^LN||{value_str}|{unit}|0.0-999.0|N|||F"


@st.composite
def oru_r01_strategy(draw: Any) -> bytes:
    """
    Generate a well-formed ORU^R01 HL7v2 message as bytes.

    Structure:
        MSH — message header
        PID — patient identification
        OBR — observation request
        OBX (1..3) — observation results
    """
    patient_id = draw(_patient_id_strategy)
    msg_id = draw(_alphanum_strategy)
    order_id = draw(_alphanum_strategy)
    ts = draw(_timestamp_strategy())
    num_obx = draw(st.integers(min_value=1, max_value=3))

    msh = f"MSH|^~\\&|LAB|HOSPITAL|EHR|HOSPITAL|{ts}||ORU^R01|{msg_id}|P|2.5"
    pid = f"PID|1||{patient_id}^^^HOSPITAL||Doe^John||19800101|M"
    obr = f"OBR|1||{order_id}|26499-4^ANC^LN|||{ts}"

    obx_lines = []
    for i in range(1, num_obx + 1):
        obx_lines.append(draw(_obx_segment_strategy(i)))

    segments = [msh, pid, obr] + obx_lines
    message = "\r".join(segments) + "\r"
    return message.encode("utf-8")


# ---------------------------------------------------------------------------
# Helper: extract stable (deterministic) fields for comparison
# ---------------------------------------------------------------------------

def _stable_key(record: CanonicalRecord) -> dict[str, Any]:
    """
    Extract only the deterministic fields from a CanonicalRecord.

    Excluded (non-deterministic): record_id, source_record_id, ingested_at
    """
    p = record.payload
    key: dict[str, Any] = {
        "patient_id": record.patient_id,
        "record_type": record.record_type,
    }
    if isinstance(p, LabRecord):
        key["loinc_code"] = p.loinc_code
        key["source_code"] = p.source_code
        key["value_numeric"] = (
            round(p.value_numeric, 6) if p.value_numeric is not None else None
        )
        key["value_text"] = p.value_text
        key["unit"] = p.unit
        key["abnormal_flag"] = p.abnormal_flag
    return key


# ---------------------------------------------------------------------------
# Property 3: HL7v2 parse idempotency
# ---------------------------------------------------------------------------

_adapter = HL7v2Adapter()


@given(raw=oru_r01_strategy())
@h_settings(
    max_examples=150,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_oru_parse_idempotency(raw: bytes) -> None:
    """
    Property 3: parsing the same ORU^R01 bytes twice produces lists of
    CanonicalRecord objects that are identical in their stable fields.

    **Validates: Requirements 21.5**
    """
    first_parse = _adapter.parse(raw)
    second_parse = _adapter.parse(raw)

    assert len(first_parse) == len(second_parse), (
        f"Parse produced different number of records: "
        f"{len(first_parse)} vs {len(second_parse)}"
    )

    first_keys = [_stable_key(r) for r in first_parse]
    second_keys = [_stable_key(r) for r in second_parse]

    assert first_keys == second_keys, (
        f"Stable fields differ between two parses of the same message:\n"
        f"  first:  {first_keys}\n"
        f"  second: {second_keys}"
    )


# ---------------------------------------------------------------------------
# Unit tests for concrete ORU^R01 messages
# ---------------------------------------------------------------------------

_SINGLE_OBX_MSG = (
    "MSH|^~\\&|LAB|HOSPITAL|EHR|HOSPITAL|20240115120000||ORU^R01|MSG001|P|2.5\r"
    "PID|1||patient-abc123^^^HOSPITAL||Doe^John||19800101|M\r"
    "OBR|1||ORDER001|26499-4^ANC^LN|||20240115120000\r"
    "OBX|1|NM|26499-4^ANC^LN||2.5|10*9/L|1.8-7.5|N|||F|||20240115120000\r"
)

_MULTI_OBX_MSG = (
    "MSH|^~\\&|LAB|HOSPITAL|EHR|HOSPITAL|20240115120000||ORU^R01|MSG002|P|2.5\r"
    "PID|1||patient-xyz789^^^HOSPITAL||Smith^Jane||19750601|F\r"
    "OBR|1||ORDER002|26499-4^ANC^LN|||20240115120000\r"
    "OBX|1|NM|26499-4^ANC^LN||1.2|10*9/L|1.8-7.5|L|||F\r"
    "OBX|2|NM|6690-2^WBC^LN||4.5|10*9/L|4.0-11.0|N|||F\r"
    "OBX|3|NM|8310-5^TEMP^LN||38.5|Cel|36.5-37.5|H|||F\r"
)

_TEXT_OBX_MSG = (
    "MSH|^~\\&|LAB|HOSPITAL|EHR|HOSPITAL|20240115120000||ORU^R01|MSG003|P|2.5\r"
    "PID|1||patient-def456^^^HOSPITAL||Brown^Bob||19900315|M\r"
    "OBR|1||ORDER003|26499-4^ANC^LN|||20240115120000\r"
    "OBX|1|TX|26499-4^ANC^LN||Pending||||||F\r"
)


def test_single_obx_parse_idempotency() -> None:
    """Parsing a single-OBX ORU^R01 message twice yields identical stable fields."""
    raw = _SINGLE_OBX_MSG.encode("utf-8")
    first = _adapter.parse(raw)
    second = _adapter.parse(raw)

    assert len(first) == 1
    assert len(second) == 1
    assert _stable_key(first[0]) == _stable_key(second[0])

    # Verify specific field values
    p = first[0].payload
    assert isinstance(p, LabRecord)
    assert first[0].patient_id == "patient-abc123"
    assert first[0].record_type == "lab"
    assert p.source_code == "26499-4"
    assert p.value_numeric == pytest.approx(2.5)
    assert p.unit == "10*9/L"
    assert p.abnormal_flag == "N"


def test_multiple_obx_parse_idempotency() -> None:
    """Parsing a multi-OBX ORU^R01 message twice yields identical stable fields for all records."""
    raw = _MULTI_OBX_MSG.encode("utf-8")
    first = _adapter.parse(raw)
    second = _adapter.parse(raw)

    assert len(first) == 3
    assert len(second) == 3

    for r1, r2 in zip(first, second):
        assert _stable_key(r1) == _stable_key(r2)

    # Verify patient_id is consistent across all records
    assert all(r.patient_id == "patient-xyz789" for r in first)
    assert all(r.record_type == "lab" for r in first)


def test_text_value_obx_parse_idempotency() -> None:
    """Parsing an OBX with a text value (non-numeric) twice yields identical stable fields."""
    raw = _TEXT_OBX_MSG.encode("utf-8")
    first = _adapter.parse(raw)
    second = _adapter.parse(raw)

    assert len(first) == 1
    assert len(second) == 1
    assert _stable_key(first[0]) == _stable_key(second[0])

    p = first[0].payload
    assert isinstance(p, LabRecord)
    assert p.value_numeric is None
    assert p.value_text == "Pending"


def test_non_deterministic_fields_differ_between_parses() -> None:
    """record_id and ingested_at are non-deterministic and may differ between parses."""
    raw = _SINGLE_OBX_MSG.encode("utf-8")
    first = _adapter.parse(raw)
    second = _adapter.parse(raw)

    # record_id is a fresh UUID each time
    assert first[0].record_id != second[0].record_id
    # source_record_id is also a fresh UUID each time
    assert first[0].source_record_id != second[0].source_record_id
