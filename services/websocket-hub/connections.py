"""
OncoIDT WebSocket connection manager.

Maintains a registry of active WebSocket connections keyed by ward_id.
Handles broadcast with automatic cleanup of stale connections on send failure.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Thread-safe (asyncio) manager for ward-scoped WebSocket connections."""

    def __init__(self) -> None:
        # ward_id -> set of active WebSocket connections
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, ward_id: str, ws: WebSocket) -> None:
        """Accept and register a WebSocket connection for a ward."""
        await ws.accept()
        self._connections[ward_id].add(ws)
        logger.info("WebSocket connected: ward=%s total=%d", ward_id, len(self._connections[ward_id]))

    def disconnect(self, ward_id: str, ws: WebSocket) -> None:
        """Remove a WebSocket connection from the registry."""
        self._connections[ward_id].discard(ws)
        logger.info("WebSocket disconnected: ward=%s remaining=%d", ward_id, len(self._connections[ward_id]))

    async def broadcast(self, ward_id: str, event: dict) -> None:
        """
        Broadcast a JSON event to all connections subscribed to ward_id.
        Stale/closed connections are removed on send failure.
        """
        stale: list[WebSocket] = []
        for ws in list(self._connections.get(ward_id, set())):
            try:
                await ws.send_json(event)
            except Exception as exc:
                logger.warning("Send failed for ward=%s, removing stale connection: %s", ward_id, exc)
                stale.append(ws)

        for ws in stale:
            self.disconnect(ward_id, ws)

    def active_count(self, ward_id: str) -> int:
        return len(self._connections.get(ward_id, set()))


# Module-level singleton shared across the application
manager = ConnectionManager()
