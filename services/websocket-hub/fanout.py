"""
OncoIDT WebSocket event fan-out.

Consumes Redis Streams and broadcasts events to ward-subscribed WebSocket clients.

Streams consumed:
  - onco:inference:score_update   → event_type: score_update
  - onco:alert:generated          → event_type: alert_generated
  - onco:ward:state_change        → event_type: ward_state_change
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from shared.redis_client import consume_events

try:
    from services.websocket_hub.connections import manager
except ImportError:
    from .connections import manager  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Stream → (consumer group, event_type label)
_STREAMS: list[tuple[str, str, str]] = [
    ("onco:inference:score_update", "ws-hub-scores", "score_update"),
    ("onco:alert:generated",        "ws-hub-alerts", "alert_generated"),
    ("onco:ward:state_change",      "ws-hub-ward",   "ward_state_change"),
]


def _extract_ward_id(event_type: str, payload: dict[str, Any]) -> str | None:
    """
    Extract ward_id from a payload.  Each stream uses slightly different field names.
    Returns None if ward_id cannot be determined (event is dropped).
    """
    # Most payloads carry ward_id directly
    ward_id = payload.get("ward_id")
    if ward_id:
        return str(ward_id)

    # score_update payloads may nest ward_id inside patient context
    if event_type == "score_update":
        ward_id = payload.get("patient", {}).get("ward_id")
        if ward_id:
            return str(ward_id)

    logger.debug("No ward_id found in %s payload: %s", event_type, list(payload.keys()))
    return None


async def _consume_stream(stream: str, group: str, event_type: str) -> None:
    """Consume a single Redis Stream and fan out to WebSocket clients."""
    consumer_name = f"ws-hub-{event_type}"
    logger.info("Starting fan-out consumer: stream=%s group=%s", stream, group)

    async for _msg_id, payload in consume_events(stream, group, consumer_name):
        ward_id = _extract_ward_id(event_type, payload)
        if ward_id is None:
            continue

        event = {"event_type": event_type, "data": payload}
        await manager.broadcast(ward_id, event)


async def start_fanout_consumers() -> None:
    """
    Launch all stream consumers as concurrent asyncio tasks.
    Intended to be called from the FastAPI lifespan startup handler.
    Each consumer restarts automatically on unexpected errors.
    """
    async def _resilient(stream: str, group: str, event_type: str) -> None:
        while True:
            try:
                await _consume_stream(stream, group, event_type)
            except asyncio.CancelledError:
                logger.info("Fan-out consumer cancelled: %s", stream)
                raise
            except Exception as exc:
                logger.error("Fan-out consumer error on %s: %s — restarting in 5s", stream, exc)
                await asyncio.sleep(5)

    for stream, group, event_type in _STREAMS:
        asyncio.create_task(_resilient(stream, group, event_type), name=f"fanout-{event_type}")
