"""
OncoIDT Alert Service — Alert Routing.

Routes alerts to the appropriate clinician roles based on priority level
and publishes routed alerts to the ``onco:alert:generated`` Redis Stream.

Routing rules (Requirement 10.4):
  Critical / High → responsible physician + charge nurse
  Medium          → assigned nurse
  Low             → care team feed

Requirements: 10.4
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from services.alerts.generator import Alert, AlertPriority
from shared.redis_client import publish_event

logger = logging.getLogger(__name__)

STREAM_ALERT_GENERATED = "onco:alert:generated"

# Mapping from priority to recipient roles
_PRIORITY_ROLES: dict[AlertPriority, list[str]] = {
    "Critical": ["Clinician", "Charge_Nurse"],
    "High": ["Clinician", "Charge_Nurse"],
    "Medium": ["Charge_Nurse"],
    "Low": ["Clinician", "Charge_Nurse", "Infection_Control_Officer"],
}


def resolve_recipient_roles(priority: AlertPriority) -> list[str]:
    """
    Return the list of recipient role names for a given alert priority.

    Critical/High → responsible physician (Clinician) + charge nurse
    Medium        → assigned nurse (Charge_Nurse)
    Low           → care team feed (all roles)
    """
    return _PRIORITY_ROLES.get(priority, ["Clinician"])


def alert_to_stream_payload(alert: Alert, recipient_roles: list[str]) -> dict[str, Any]:
    """Serialise an Alert to a Redis Stream-compatible dict."""
    return {
        "alert_id": alert.alert_id,
        "patient_id": alert.patient_id or "",
        "ward_id": alert.ward_id or "",
        "alert_type": alert.alert_type,
        "priority": alert.priority,
        "score": str(alert.score) if alert.score is not None else "",
        "score_delta": str(alert.score_delta) if alert.score_delta is not None else "",
        "message": alert.message,
        "escalation_count": str(alert.escalation_count),
        "generated_at": alert.generated_at.isoformat(),
        "recipient_roles": ",".join(recipient_roles),
        "top_features_count": str(len(alert.top_features)),
        "is_duplicate": "false",
    }


async def route_alert(
    alert: Alert,
    redis: Redis,
    is_duplicate: bool = False,
) -> list[str]:
    """
    Determine recipient roles for the alert and publish to ``onco:alert:generated``.

    Parameters
    ----------
    alert:
        The Alert to route (may have escalation_count > 0 if deduplicated).
    redis:
        Async Redis client.
    is_duplicate:
        Whether this alert was deduplicated (escalation_count was incremented).

    Returns
    -------
    List of recipient role strings that were assigned.
    """
    recipient_roles = resolve_recipient_roles(alert.priority)

    payload = alert_to_stream_payload(alert, recipient_roles)
    payload["is_duplicate"] = "true" if is_duplicate else "false"

    await publish_event(STREAM_ALERT_GENERATED, payload, redis=redis)

    logger.info(
        "Alert routed: id=%s type=%s priority=%s patient=%s roles=%s escalation=%d",
        alert.alert_id,
        alert.alert_type,
        alert.priority,
        alert.patient_id,
        recipient_roles,
        alert.escalation_count,
    )
    return recipient_roles
