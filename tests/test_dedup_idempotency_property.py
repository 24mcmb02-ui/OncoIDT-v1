"""
Property test — Ingestion deduplication idempotency.

Property 4 (Requirement 3.4):
    Ingesting the same CanonicalRecord twice results in exactly one row in
    clinical_events with the same record_id.

The test runs entirely in-memory using a fake async session backed by a plain
Python list, so it requires no database connection and runs fast under Hypothesis.

**Validates: Requirements 3.4**
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings as h_settings
from hypothesis import strategies as st

from services.ingestion.dedup import (
    DedupResult,
    compute_dedup_hash,
    deduplicate,
)
from shared.models import (
    CanonicalRecord,
    DataQualityFlag,
    LabRecord,
    VitalRecord,
)


# ---------------------------------------------------------------------------
# In-memory fake async session
# ---------------------------------------------------------------------------

class _FakeRow:
    def __init__(self, *values: Any) -> None:
        self._values = values

    def __getitem__(self, idx: int) -> Any:
        return self._values[idx]


class _FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchone(self) -> _FakeRow | None:
        return _FakeRow(*self._rows[0]) if self._rows else None

    def fetchall(self) -> list[_FakeRow]:
        return [_FakeRow(*r) for r in self._rows]


class InMemoryClinicalEventsSession:
    """
    Fake AsyncSession that stores clinical_events rows in a plain Python list.
    Supports the exact SQL patterns used by services/ingestion/dedup.py.
    """

    def __init__(self) -> None:
        # Each entry: {"record_id": str, "_dedup_hash": str}
        self._rows: list[dict[str, Any]] = []
        # data_quality_flags rows
        self._flags: list[dict[str, Any]] = []

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        sql = str(stmt).strip()

        # Exact-hash lookup
        if "payload->>'_dedup_hash' = :hash" in sql and params:
            target = params["hash"]
            matches = [r for r in self._rows if r["_dedup_hash"] == target]
            return _FakeResult([(r["record_id"],) for r in matches])

        # Near-duplicate prefix lookup
        if "payload->>'_dedup_hash' LIKE :prefix" in sql and params:
            prefix = params["prefix"].rstrip("%")
            matches = [r for r in self._rows if r["_dedup_hash"].startswith(prefix)]
            return _FakeResult([(r["record_id"],) for r in matches])

        # data_quality_flags INSERT
        if "INSERT INTO data_quality_flags" in sql and params:
            self._flags.append(dict(params))
            return _FakeResult([])

        raise NotImplementedError(f"Unhandled SQL in fake session:\n{sql}")

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    def persist_record(self, record: CanonicalRecord, dedup_hash: str) -> None:
        """Simulate the caller persisting an ACCEPTED record to clinical_events."""
        self._rows.append({"record_id": record.record_id, "_dedup_hash": dedup_hash})

    def count_by_record_id(self, record_id: str) -> int:
        return sum(1 for r in self._rows if r["record_id"] == record_id)

    def count_by_hash(self, dedup_hash: str) -> int:
        return sum(1 for r in self._rows if r["_dedup_hash"] == dedup_hash)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_lab_record(
    patient_id: str,
    source_system: str,
    timestamp_utc: datetime,
    loinc_code: str = "26499-4",
    value: float = 2.5,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system=source_system,
        source_record_id=str(uuid.uuid4()),
        record_type="lab",
        timestamp_utc=timestamp_utc,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=LabRecord(
            loinc_code=loinc_code,
            source_code=loinc_code,
            source_system_code=source_system,
            value_numeric=value,
            value_text=None,
            unit="10*9/L",
            reference_range_low=1.8,
            reference_range_high=7.5,
            abnormal_flag="N",
        ),
    )


def _make_vital_record(
    patient_id: str,
    source_system: str,
    timestamp_utc: datetime,
    loinc_code: str = "8310-5",
    value: float = 37.2,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system=source_system,
        source_record_id=str(uuid.uuid4()),
        record_type="vital",
        timestamp_utc=timestamp_utc,
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=VitalRecord(
            loinc_code=loinc_code,
            value_numeric=value,
            value_text=None,
            unit="Cel",
        ),
    )


async def _ingest(session: InMemoryClinicalEventsSession, record: CanonicalRecord) -> DedupResult:
    """
    Simulate the ingestion pipeline for a single record:
      1. Run deduplication check.
      2. If ACCEPTED or NEAR_DUPLICATE, persist to clinical_events.
      3. Return the DedupResult.
    """
    result, dedup_hash = await deduplicate(session, record)  # type: ignore[arg-type]
    if result in (DedupResult.ACCEPTED, DedupResult.NEAR_DUPLICATE):
        session.persist_record(record, dedup_hash)
    return result


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_patient_id_st = st.from_regex(r"patient-[a-z0-9]{6}", fullmatch=True)
_source_system_st = st.sampled_from(["EHR-A", "EHR-B", "LAB-SYS", "MONITOR"])
_loinc_st = st.sampled_from(["26499-4", "6690-2", "8310-5", "731-0", "1988-5"])
_value_st = st.floats(min_value=0.01, max_value=999.0, allow_nan=False, allow_infinity=False)
_ts_st = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2026, 1, 1),
    timezones=st.just(timezone.utc),
)
_record_type_st = st.sampled_from(["lab", "vital"])


@st.composite
def canonical_record_strategy(draw: Any) -> CanonicalRecord:
    patient_id = draw(_patient_id_st)
    source_system = draw(_source_system_st)
    ts = draw(_ts_st)
    loinc = draw(_loinc_st)
    value = draw(_value_st)
    record_type = draw(_record_type_st)

    if record_type == "lab":
        return _make_lab_record(patient_id, source_system, ts, loinc, value)
    return _make_vital_record(patient_id, source_system, ts, loinc, value)


# ---------------------------------------------------------------------------
# Property 4: Deduplication idempotency
# ---------------------------------------------------------------------------

@given(record=canonical_record_strategy())
@h_settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_double_ingest_produces_one_row(record: CanonicalRecord) -> None:
    """
    Property 4: Ingesting the same CanonicalRecord twice results in exactly
    one row in clinical_events with the same record_id.

    The second ingest must be detected as EXACT_DUPLICATE and discarded.

    **Validates: Requirements 3.4**
    """
    session = InMemoryClinicalEventsSession()

    # First ingest — must be accepted
    first_result = _run(_ingest(session, record))
    assert first_result == DedupResult.ACCEPTED, (
        f"First ingest should be ACCEPTED, got {first_result}"
    )

    # Second ingest of the identical record — must be detected as exact duplicate
    second_result = _run(_ingest(session, record))
    assert second_result == DedupResult.EXACT_DUPLICATE, (
        f"Second ingest should be EXACT_DUPLICATE, got {second_result}"
    )

    # Exactly one row must exist in clinical_events
    row_count = session.count_by_record_id(record.record_id)
    assert row_count == 1, (
        f"Expected exactly 1 row in clinical_events for record_id={record.record_id}, "
        f"found {row_count}"
    )


@given(
    record=canonical_record_strategy(),
    n=st.integers(min_value=3, max_value=10),
)
@h_settings(
    max_examples=150,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_n_ingests_of_same_record_produces_one_row(record: CanonicalRecord, n: int) -> None:
    """
    Property 4 (extended): Ingesting the same CanonicalRecord N times always
    results in exactly one row in clinical_events.

    **Validates: Requirements 3.4**
    """
    session = InMemoryClinicalEventsSession()

    results = [_run(_ingest(session, record)) for _ in range(n)]

    # First must be ACCEPTED; all subsequent must be EXACT_DUPLICATE
    assert results[0] == DedupResult.ACCEPTED
    assert all(r == DedupResult.EXACT_DUPLICATE for r in results[1:]), (
        f"Expected all ingests after the first to be EXACT_DUPLICATE, got: {results[1:]}"
    )

    # Still exactly one row
    dedup_hash = compute_dedup_hash(record)
    assert session.count_by_hash(dedup_hash) == 1, (
        f"Expected 1 row for hash={dedup_hash[:16]}..., "
        f"found {session.count_by_hash(dedup_hash)}"
    )


@given(
    record_a=canonical_record_strategy(),
    record_b=canonical_record_strategy(),
)
@h_settings(
    max_examples=150,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_distinct_records_both_persisted(
    record_a: CanonicalRecord, record_b: CanonicalRecord
) -> None:
    """
    Two records with different identity tuples must both be persisted
    (no false-positive deduplication).

    **Validates: Requirements 3.4**
    """
    hash_a = compute_dedup_hash(record_a)
    hash_b = compute_dedup_hash(record_b)

    # Only test the property when the hashes are genuinely different
    # (same hash would mean same identity tuple — a valid collision in the strategy)
    if hash_a == hash_b:
        return

    session = InMemoryClinicalEventsSession()

    result_a = _run(_ingest(session, record_a))
    result_b = _run(_ingest(session, record_b))

    assert result_a == DedupResult.ACCEPTED
    assert result_b == DedupResult.ACCEPTED

    assert session.count_by_hash(hash_a) == 1
    assert session.count_by_hash(hash_b) == 1
    assert len(session._rows) == 2


# ---------------------------------------------------------------------------
# Unit tests for concrete scenarios
# ---------------------------------------------------------------------------

def test_exact_duplicate_discarded() -> None:
    """Concrete: ingesting the same lab record twice keeps exactly one row."""
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    record = _make_lab_record("patient-abc123", "EHR-A", ts)

    session = InMemoryClinicalEventsSession()
    r1 = _run(_ingest(session, record))
    r2 = _run(_ingest(session, record))

    assert r1 == DedupResult.ACCEPTED
    assert r2 == DedupResult.EXACT_DUPLICATE
    assert len(session._rows) == 1
    assert session._rows[0]["record_id"] == record.record_id


def test_different_patients_not_deduplicated() -> None:
    """Two records for different patients with the same timestamp are both kept."""
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-aaa111", "EHR-A", ts)
    r2 = _make_lab_record("patient-bbb222", "EHR-A", ts)

    session = InMemoryClinicalEventsSession()
    assert _run(_ingest(session, r1)) == DedupResult.ACCEPTED
    assert _run(_ingest(session, r2)) == DedupResult.ACCEPTED
    assert len(session._rows) == 2


def test_different_source_systems_not_deduplicated() -> None:
    """Same patient + timestamp from two different source systems are both kept."""
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-abc123", "EHR-A", ts)
    r2 = _make_lab_record("patient-abc123", "EHR-B", ts)

    session = InMemoryClinicalEventsSession()
    assert _run(_ingest(session, r1)) == DedupResult.ACCEPTED
    assert _run(_ingest(session, r2)) == DedupResult.ACCEPTED
    assert len(session._rows) == 2


def test_different_timestamps_not_deduplicated() -> None:
    """Same patient + source but different timestamps are both kept."""
    ts1 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-abc123", "EHR-A", ts1)
    r2 = _make_lab_record("patient-abc123", "EHR-A", ts2)

    session = InMemoryClinicalEventsSession()
    assert _run(_ingest(session, r1)) == DedupResult.ACCEPTED
    assert _run(_ingest(session, r2)) == DedupResult.ACCEPTED
    assert len(session._rows) == 2


def test_dedup_hash_is_deterministic() -> None:
    """compute_dedup_hash returns the same value for the same record identity."""
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    record = _make_lab_record("patient-abc123", "EHR-A", ts)
    assert compute_dedup_hash(record) == compute_dedup_hash(record)


def test_dedup_hash_changes_on_patient_id() -> None:
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-aaa111", "EHR-A", ts)
    r2 = _make_lab_record("patient-bbb222", "EHR-A", ts)
    assert compute_dedup_hash(r1) != compute_dedup_hash(r2)


def test_dedup_hash_changes_on_source_system() -> None:
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-abc123", "EHR-A", ts)
    r2 = _make_lab_record("patient-abc123", "EHR-B", ts)
    assert compute_dedup_hash(r1) != compute_dedup_hash(r2)


def test_dedup_hash_changes_on_timestamp() -> None:
    ts1 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 15, 12, 0, 1, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-abc123", "EHR-A", ts1)
    r2 = _make_lab_record("patient-abc123", "EHR-A", ts2)
    assert compute_dedup_hash(r1) != compute_dedup_hash(r2)


def test_dedup_hash_changes_on_record_type() -> None:
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    r1 = _make_lab_record("patient-abc123", "EHR-A", ts)
    r2 = _make_vital_record("patient-abc123", "EHR-A", ts)
    # record_type differs ("lab" vs "vital") so hashes must differ
    assert compute_dedup_hash(r1) != compute_dedup_hash(r2)
