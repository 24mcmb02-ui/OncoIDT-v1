"""
OncoIDT Alert Service — Alert Actions and Snooze Re-evaluation.

Implements acknowledge, snooze, and escalate actions on alerts.
Writes action records to the ``alert_actions`` table and publishes
events to the ``onco:alert:action`` Redis Stream.

Snooze re-evaluation: schedules a background task that re-evaluates the
triggering condition at snooze expiry and re-alerts if the condition persists.

Requirements: 10.5, 10.6
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.audit import append_audit_entry
from shared.redis_client import publish_event

logger = logging.getLogger(__name__)

STREAM_ALERT_ACTION = "onco:alert:action"
ActionType = Literal["acknowledge", "snooze", "escalate"]

# In-process registry of pending snooze re-evaluation tasks
# Maps alert_id → asyncio.Task
_snooze_tasks: dict[str, asyncio.Task[None]] = {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _insert_alert_action(
    session: AsyncSession,
    alert_id: str,
    action_type: ActionType,
    user_id: str,
    reason: str | None,
    snooze_until: datetime | None,
) -> str:
    """Insert a row into alert_actions and return the action_id."""
    action_id = str(uuid.uuid4())
    await session.execute(
        text(
            """
            INSERT INTO alert_actions
                (action_id, alert_id, action_type, user_id, reason, snooze_until, created_at)
            VALUES
                (:action_id, :alert_id, :action_type, :user_id, :reason, :snooze_until, :created_at)
            """
        ),
        {
            "action_id": action_id,
            "alert_id": alert_id,
            "action_type": action_type,
            "user_id": user_id,
            "reason": reason,
            "snooze_until": snooze_until,
            "created_at": datetime.now(timezone.utc),
        },
    )
    return action_id


async def _update_alert_status(
    session: AsyncSession,
    alert_id: str,
    acknowledged: bool | None = None,
    snoozed_until: datetime | None = None,
    priority: str | None = None,
) -> None:
    """Update mutable fields on an alert row."""
    updates: list[str] = []
    params: dict[str, Any] = {"alert_id": alert_id}

    if acknowledged is not None:
        updates.append("acknowledged = :acknowledged")
        params["acknowledged"] = acknowledged
    if snoozed_until is not None:
        updates.append("snoozed_until = :snoozed_until")
        params["snoozed_until"] = snoozed_until
    if priority is not None:
        updates.append("priority = :priority")
        params["priority"] = priority

    if not updates:
        return

    await session.execute(
        text(f"UPDATE alerts SET {', '.join(updates)} WHERE alert_id = :alert_id"),
        params,
    )


async def _fetch_alert(session: AsyncSession, alert_id: str) -> dict[str, Any] | None:
    """Fetch a single alert row by ID."""
    result = await session.execute(
        text("SELECT * FROM alerts WHERE alert_id = :alert_id"),
        {"alert_id": alert_id},
    )
    row = result.mappings().fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

async def acknowledge(
    alert_id: str,
    user_id: str,
    session: AsyncSession,
    redis: Redis,
) -> str:
    """
    Acknowledge an alert.

    Marks the alert as acknowledged in the DB, writes an alert_actions record,
    and publishes to onco:alert:action.

    Returns the action_id.
    """
    action_id = await _insert_alert_action(
        session, alert_id, "acknowledge", user_id, reason=None, snooze_until=None
    )
    await _update_alert_status(session, alert_id, acknowledged=True)
    await append_audit_entry(
        session,
        user_id=user_id,
        action="alert_acknowledge",
        resource_type="alert",
        resource_id=alert_id,
        details={"action_id": action_id},
    )
    await session.commit()

    await publish_event(
        STREAM_ALERT_ACTION,
        {
            "action_id": action_id,
            "alert_id": alert_id,
            "action_type": "acknowledge",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        redis=redis,
    )
    logger.info("Alert %s acknowledged by user %s", alert_id, user_id)
    return action_id


async def snooze(
    alert_id: str,
    user_id: str,
    reason: str,
    duration_minutes: int,
    session: AsyncSession,
    redis: Redis,
    reevaluate_callback: Any | None = None,
) -> str:
    """
    Snooze an alert for ``duration_minutes``.

    Writes to alert_actions, updates the alert's snoozed_until timestamp,
    publishes to onco:alert:action, and schedules a re-evaluation task that
    fires at snooze expiry (Requirement 10.6).

    Parameters
    ----------
    reevaluate_callback:
        Optional async callable(alert_id) invoked at snooze expiry to
        re-evaluate the triggering condition. If None, only the stream
        event is published at expiry.

    Returns the action_id.
    """
    snooze_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

    action_id = await _insert_alert_action(
        session, alert_id, "snooze", user_id, reason=reason, snooze_until=snooze_until
    )
    await _update_alert_status(session, alert_id, snoozed_until=snooze_until)
    await append_audit_entry(
        session,
        user_id=user_id,
        action="alert_snooze",
        resource_type="alert",
        resource_id=alert_id,
        details={
            "action_id": action_id,
            "reason": reason,
            "duration_minutes": duration_minutes,
            "snooze_until": snooze_until.isoformat(),
        },
    )
    await session.commit()

    await publish_event(
        STREAM_ALERT_ACTION,
        {
            "action_id": action_id,
            "alert_id": alert_id,
            "action_type": "snooze",
            "user_id": user_id,
            "reason": reason,
            "duration_minutes": duration_minutes,
            "snooze_until": snooze_until.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        redis=redis,
    )

    # Schedule re-evaluation at snooze expiry
    _cancel_snooze_task(alert_id)
    task = asyncio.create_task(
        _snooze_reeval_task(
            alert_id=alert_id,
            delay_seconds=duration_minutes * 60,
            redis=redis,
            callback=reevaluate_callback,
        ),
        name=f"snooze-reeval-{alert_id}",
    )
    _snooze_tasks[alert_id] = task

    logger.info(
        "Alert %s snoozed by user %s for %d min (until %s)",
        alert_id, user_id, duration_minutes, snooze_until.isoformat(),
    )
    return action_id


async def escalate(
    alert_id: str,
    user_id: str,
    session: AsyncSession,
    redis: Redis,
) -> str:
    """
    Escalate an alert to Critical priority.

    Writes to alert_actions, upgrades priority in DB, publishes to
    onco:alert:action.

    Returns the action_id.
    """
    action_id = await _insert_alert_action(
        session, alert_id, "escalate", user_id, reason=None, snooze_until=None
    )
    await _update_alert_status(session, alert_id, priority="Critical")
    await append_audit_entry(
        session,
        user_id=user_id,
        action="alert_escalate",
        resource_type="alert",
        resource_id=alert_id,
        details={"action_id": action_id},
    )
    await session.commit()

    await publish_event(
        STREAM_ALERT_ACTION,
        {
            "action_id": action_id,
            "alert_id": alert_id,
            "action_type": "escalate",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        redis=redis,
    )
    logger.info("Alert %s escalated by user %s", alert_id, user_id)
    return action_id


# ---------------------------------------------------------------------------
# Snooze re-evaluation background task
# ---------------------------------------------------------------------------

async def _snooze_reeval_task(
    alert_id: str,
    delay_seconds: int,
    redis: Redis,
    callback: Any | None,
) -> None:
    """
    Wait for the snooze duration, then re-evaluate the alert condition.
    If the callback indicates the condition persists, publish a re-alert event.
    """
    try:
        await asyncio.sleep(delay_seconds)
    except asyncio.CancelledError:
        logger.debug("Snooze re-eval task cancelled for alert %s", alert_id)
        return

    logger.info("Snooze expired for alert %s — re-evaluating condition", alert_id)

    condition_persists = True
    if callback is not None:
        try:
            condition_persists = await callback(alert_id)
        except Exception:
            logger.exception("Re-evaluation callback failed for alert %s", alert_id)
            condition_persists = True  # conservative: re-alert on callback failure

    if condition_persists:
        await publish_event(
            STREAM_ALERT_ACTION,
            {
                "alert_id": alert_id,
                "action_type": "snooze_expired_realert",
                "condition_persists": "true",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            redis=redis,
        )
        logger.info("Re-alert published for alert %s after snooze expiry", alert_id)
    else:
        logger.info("Condition resolved for alert %s — no re-alert needed", alert_id)

    _snooze_tasks.pop(alert_id, None)


def _cancel_snooze_task(alert_id: str) -> None:
    """Cancel any existing snooze re-evaluation task for this alert."""
    existing = _snooze_tasks.pop(alert_id, None)
    if existing and not existing.done():
        existing.cancel()


def cancel_all_snooze_tasks() -> None:
    """Cancel all pending snooze tasks — call on service shutdown."""
    for alert_id, task in list(_snooze_tasks.items()):
        if not task.done():
            task.cancel()
    _snooze_tasks.clear()
