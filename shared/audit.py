"""
OncoIDT audit log — append-only with SHA-256 hash chain.

Every entry's `entry_hash` is:
    SHA-256(prev_hash || timestamp || user_id || action || resource_id || details_json)

This makes any post-hoc mutation of a row detectable because all subsequent
hashes will no longer validate.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Sentinel hash used for the very first entry (no predecessor)
GENESIS_HASH = "0" * 64


def _compute_hash(
    prev_hash: str,
    timestamp: datetime,
    user_id: str,
    action: str,
    resource_id: str,
    details: dict[str, Any] | None,
) -> str:
    """Compute SHA-256 hash for an audit entry."""
    details_json = json.dumps(details, sort_keys=True, default=str) if details else "{}"
    raw = f"{prev_hash}{timestamp.isoformat()}{user_id}{action}{resource_id}{details_json}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _get_last_hash(session: AsyncSession) -> str:
    """Retrieve the entry_hash of the most recent audit log row."""
    result = await session.execute(
        text("SELECT entry_hash FROM audit_log ORDER BY entry_id DESC LIMIT 1")
    )
    row = result.fetchone()
    return row[0] if row else GENESIS_HASH


async def append_audit_entry(
    session: AsyncSession,
    *,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> int:
    """
    Append a tamper-evident entry to the audit_log table.

    Returns the new entry_id.

    The caller is responsible for committing the session (or the session
    dependency will commit on request completion).
    """
    ts = timestamp or datetime.now(timezone.utc)
    prev_hash = await _get_last_hash(session)
    entry_hash = _compute_hash(prev_hash, ts, user_id, action, resource_id, details)

    result = await session.execute(
        text(
            """
            INSERT INTO audit_log
                (timestamp, user_id, action, resource_type, resource_id, details, prev_hash, entry_hash)
            VALUES
                (:timestamp, :user_id, :action, :resource_type, :resource_id, :details, :prev_hash, :entry_hash)
            RETURNING entry_id
            """
        ),
        {
            "timestamp": ts,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": json.dumps(details, default=str) if details else None,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        },
    )
    entry_id: int = result.scalar_one()
    logger.debug("Audit entry %d written: %s %s %s", entry_id, user_id, action, resource_id)
    return entry_id


async def verify_audit_chain(session: AsyncSession) -> bool:
    """
    Walk the entire audit log and verify the hash chain is intact.
    Returns True if valid, False if any entry has been tampered with.
    """
    result = await session.execute(
        text(
            """
            SELECT entry_id, timestamp, user_id, action, resource_id, details, prev_hash, entry_hash
            FROM audit_log
            ORDER BY entry_id ASC
            """
        )
    )
    rows = result.fetchall()

    running_hash = GENESIS_HASH
    for row in rows:
        entry_id, ts, user_id, action, resource_id, details_raw, prev_hash, stored_hash = row

        if prev_hash != running_hash:
            logger.warning("Audit chain broken at entry_id=%d: prev_hash mismatch", entry_id)
            return False

        details = json.loads(details_raw) if details_raw else None
        expected_hash = _compute_hash(prev_hash, ts, user_id, action, resource_id, details)

        if expected_hash != stored_hash:
            logger.warning("Audit chain broken at entry_id=%d: entry_hash mismatch", entry_id)
            return False

        running_hash = stored_hash

    return True
