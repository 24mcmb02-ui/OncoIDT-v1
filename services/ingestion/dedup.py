"""
Deduplication logic for the OncoIDT ingestion service.

Strategy:
  - Compute SHA-256 hash of (patient_id, source_system, timestamp_utc, record_type)
  - Query clinical_events for an existing row with the same dedup_hash
  - Exact duplicates (same hash) → discard silently
  - Near-duplicates (same hash prefix but different payload) → log to data_quality_flags

The dedup_hash is stored in clinical_events.payload JSONB under the key "_dedup_hash"
so it can be indexed and queried without a schema change.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.models import CanonicalRecord, DataQualityFlag

logger = logging.getLogger(__name__)


class DedupResult(Enum):
    ACCEPTED = "accepted"       # New record, should be persisted
    EXACT_DUPLICATE = "exact_duplicate"   # Identical hash — discard
    NEAR_DUPLICATE = "near_duplicate"     # Same hash prefix, different payload — flag and accept


def compute_dedup_hash(record: CanonicalRecord) -> str:
    """
    Compute a deterministic SHA-256 hash of the record's identity tuple:
        (patient_id, source_system, timestamp_utc_iso, record_type)
    """
    ts_iso = record.timestamp_utc.astimezone(timezone.utc).isoformat()
    raw = f"{record.patient_id}|{record.source_system}|{ts_iso}|{record.record_type}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def check_duplicate(
    session: AsyncSession,
    record: CanonicalRecord,
    dedup_hash: str,
) -> DedupResult:
    """
    Query clinical_events for an existing row with the same dedup_hash.

    Returns:
        EXACT_DUPLICATE  — a row with this hash already exists
        NEAR_DUPLICATE   — a row with the same 16-char hash prefix exists (different full hash)
        ACCEPTED         — no matching row found
    """
    # Check for exact match
    result = await session.execute(
        text(
            "SELECT record_id FROM clinical_events "
            "WHERE payload->>'_dedup_hash' = :hash LIMIT 1"
        ),
        {"hash": dedup_hash},
    )
    if result.fetchone() is not None:
        logger.debug("Exact duplicate discarded: hash=%s patient=%s", dedup_hash[:16], record.patient_id)
        return DedupResult.EXACT_DUPLICATE

    # Check for near-duplicate (same 16-char prefix)
    prefix = dedup_hash[:16]
    result = await session.execute(
        text(
            "SELECT record_id FROM clinical_events "
            "WHERE payload->>'_dedup_hash' LIKE :prefix LIMIT 1"
        ),
        {"prefix": f"{prefix}%"},
    )
    if result.fetchone() is not None:
        logger.warning(
            "Near-duplicate detected: hash_prefix=%s patient=%s record_type=%s",
            prefix, record.patient_id, record.record_type,
        )
        return DedupResult.NEAR_DUPLICATE

    return DedupResult.ACCEPTED


async def log_near_duplicate(
    session: AsyncSession,
    record: CanonicalRecord,
    dedup_hash: str,
) -> None:
    """
    Insert a row into data_quality_flags for a near-duplicate record.
    """
    await session.execute(
        text(
            """
            INSERT INTO data_quality_flags
                (record_id, patient_id, flag_type, field_name, message, severity, created_at)
            VALUES
                (:record_id, :patient_id, :flag_type, :field_name, :message, :severity, :created_at)
            """
        ),
        {
            "record_id": record.record_id,
            "patient_id": record.patient_id,
            "flag_type": "near_duplicate",
            "field_name": "dedup_hash",
            "message": (
                f"Near-duplicate detected for patient={record.patient_id} "
                f"source={record.source_system} ts={record.timestamp_utc.isoformat()} "
                f"type={record.record_type} hash_prefix={dedup_hash[:16]}"
            ),
            "severity": "warning",
            "created_at": datetime.now(timezone.utc),
        },
    )


async def deduplicate(
    session: AsyncSession,
    record: CanonicalRecord,
) -> tuple[DedupResult, str]:
    """
    Full deduplication pipeline for a single CanonicalRecord.

    Returns (DedupResult, dedup_hash).
    Callers should:
      - Discard the record if result is EXACT_DUPLICATE
      - Persist the record (with quality flag) if NEAR_DUPLICATE
      - Persist normally if ACCEPTED
    """
    dedup_hash = compute_dedup_hash(record)
    result = await check_duplicate(session, record, dedup_hash)

    if result == DedupResult.NEAR_DUPLICATE:
        await log_near_duplicate(session, record, dedup_hash)
        # Also attach the flag to the in-memory record
        record.data_quality_flags.append(
            DataQualityFlag(
                flag_type="near_duplicate",
                field_name="dedup_hash",
                message=f"Near-duplicate of existing record (hash_prefix={dedup_hash[:16]})",
                severity="warning",
            )
        )

    return result, dedup_hash
