"""
OncoIDT Alert Service — Redis-Backed Alert Deduplication.

Deduplicates alerts within a 30-minute window per (patient_id, alert_type) key.
When a duplicate is detected, the existing alert's timestamp is updated and
its escalation_count is incremented rather than creating a new alert record.

Requirements: 10.3
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from services.alerts.generator import Alert
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Redis key pattern: onco:alert:dedup:{patient_id}:{alert_type}
_KEY_PREFIX = "onco:alert:dedup"


def _dedup_key(patient_id: str | None, alert_type: str) -> str:
    pid = patient_id or "ward"
    return f"{_KEY_PREFIX}:{pid}:{alert_type}"


async def deduplicate_alert(
    alert: Alert,
    redis: Redis,
    window_minutes: int | None = None,
) -> tuple[Alert, bool]:
    """
    Check whether an equivalent alert already exists within the dedup window.

    Parameters
    ----------
    alert:
        The newly generated Alert to check.
    redis:
        Async Redis client.
    window_minutes:
        Deduplication window in minutes. Defaults to settings.alert_dedup_window_minutes (30).

    Returns
    -------
    (alert, is_duplicate):
        - If not a duplicate: returns the original alert unchanged, is_duplicate=False.
        - If a duplicate: returns the updated alert (incremented escalation_count,
          refreshed timestamp), is_duplicate=True.
          The caller should UPDATE the existing record rather than INSERT a new one.
    """
    window = window_minutes or settings.alert_dedup_window_minutes
    window_seconds = window * 60
    key = _dedup_key(alert.patient_id, alert.alert_type)

    existing_raw: str | None = await redis.get(key)

    if existing_raw is None:
        # No existing alert — store this one and return as new
        entry = {
            "alert_id": alert.alert_id,
            "escalation_count": 0,
            "first_generated_at": alert.generated_at.isoformat(),
            "score": alert.score,
        }
        await redis.setex(key, window_seconds, json.dumps(entry))
        return alert, False

    # Duplicate found — update escalation count and timestamp
    try:
        existing = json.loads(existing_raw)
    except (json.JSONDecodeError, TypeError):
        existing = {}

    escalation_count = int(existing.get("escalation_count", 0)) + 1
    original_alert_id: str = existing.get("alert_id", alert.alert_id)

    # Update the Redis entry with new count and refresh TTL
    updated_entry = {
        "alert_id": original_alert_id,
        "escalation_count": escalation_count,
        "first_generated_at": existing.get("first_generated_at", alert.generated_at.isoformat()),
        "score": alert.score,
    }
    await redis.setex(key, window_seconds, json.dumps(updated_entry))

    # Return a mutated copy of the alert reflecting the dedup state
    alert.alert_id = original_alert_id
    alert.escalation_count = escalation_count
    alert.generated_at = datetime.now(timezone.utc)

    logger.debug(
        "Deduplicated alert type=%s patient=%s escalation_count=%d",
        alert.alert_type,
        alert.patient_id,
        escalation_count,
    )
    return alert, True


async def clear_dedup_entry(
    patient_id: str | None,
    alert_type: str,
    redis: Redis,
) -> None:
    """
    Remove the deduplication entry for a (patient_id, alert_type) pair.
    Called when an alert is acknowledged or resolved to reset the window.
    """
    key = _dedup_key(patient_id, alert_type)
    await redis.delete(key)
    logger.debug("Cleared dedup entry for patient=%s type=%s", patient_id, alert_type)


async def get_dedup_state(
    patient_id: str | None,
    alert_type: str,
    redis: Redis,
) -> dict[str, Any] | None:
    """
    Return the current dedup state for a (patient_id, alert_type) pair, or None.
    Useful for inspecting escalation_count without triggering dedup logic.
    """
    key = _dedup_key(patient_id, alert_type)
    raw: str | None = await redis.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
