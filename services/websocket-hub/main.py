"""
OncoIDT WebSocket Hub service.

Endpoint:
  WS /ws/v1/ward/{ward_id}?token=<jwt>

Clients authenticate via JWT query parameter, then receive real-time push
events for the subscribed ward (score_update, alert_generated, ward_state_change).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, status

from shared.auth import _decode_token
from shared.config import get_settings
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool

try:
    # Absolute imports (Docker: copied as services.websocket_hub)
    from services.websocket_hub.connections import manager
    from services.websocket_hub.fanout import start_fanout_consumers
except ImportError:
    # Relative imports (local dev: services/websocket-hub/)
    from .connections import manager  # type: ignore[no-redef]
    from .fanout import start_fanout_consumers  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("WebSocket hub starting — launching fan-out consumers")
    await start_fanout_consumers()
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    yield
    logger.info("WebSocket hub shutting down")
    await graceful_shutdown([error_monitor_task])
    await close_redis_pool()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT WebSocket Hub",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(make_health_router(service_name="websocket-hub"))


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/v1/ward/{ward_id}")
async def ward_websocket(
    ward_id: str,
    ws: WebSocket,
    token: str = Query(..., description="JWT access token"),
) -> None:
    """
    Authenticate the client via JWT query param, then stream ward events.
    The connection is closed with 4001 if authentication fails or the user
    is not authorised for the requested ward.
    """
    # Authenticate
    try:
        user = _decode_token(token)
    except Exception as exc:
        logger.warning("WebSocket auth failed for ward=%s: %s", ward_id, exc)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Authorise ward access
    if not user.can_access_ward(ward_id):
        logger.warning("User %s not authorised for ward=%s", user.user_id, ward_id)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(ward_id, ws)
    try:
        # Keep connection alive; client drives disconnection
        while True:
            # Drain any client-sent messages (ping/pong or graceful close)
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ward_id, ws)
