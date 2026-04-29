"""
OncoIDT shared Prometheus metrics instrumentation.

Usage:
    from shared.metrics import setup_metrics, start_error_rate_monitor, graceful_shutdown

    # At app creation:
    setup_metrics(app, settings.service_name)

    # In lifespan startup:
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)

    # In lifespan shutdown:
    await graceful_shutdown([error_monitor_task, ...])
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level metric singletons (avoid duplicate registration)
# ---------------------------------------------------------------------------

_metrics_registry: dict[str, Any] = {}


def _get_or_create(name: str, factory):  # type: ignore[type-arg]
    """Return existing metric or create via factory (singleton pattern)."""
    if name not in _metrics_registry:
        _metrics_registry[name] = factory()
    return _metrics_registry[name]


# ---------------------------------------------------------------------------
# ServiceMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ServiceMetrics:
    inference_latency: Any = field(default=None)
    active_patients: Any = field(default=None)
    alert_generated: Any = field(default=None)
    score_staleness: Any = field(default=None)
    redis_stream_lag: Any = field(default=None)


# ---------------------------------------------------------------------------
# setup_metrics
# ---------------------------------------------------------------------------

def setup_metrics(app: FastAPI, service_name: str) -> ServiceMetrics:
    """
    Instrument the FastAPI app with Prometheus metrics.

    - Mounts prometheus_fastapi_instrumentator on /metrics
    - Registers custom OncoIDT metrics as singletons
    - Registers error-tracking middleware for the error rate monitor

    Falls back gracefully if prometheus_client or
    prometheus_fastapi_instrumentator are not installed.
    """
    metrics = ServiceMetrics()

    # --- Error tracking middleware (must be registered before app starts) ----
    @app.middleware("http")
    async def _error_tracking_middleware(request: Request, call_next) -> Response:  # type: ignore[type-arg]
        response = await call_next(request)
        now = datetime.now(timezone.utc).timestamp()
        is_error = response.status_code >= 500
        _error_window.append((now, is_error))
        return response

    # --- HTTP instrumentation via prometheus_fastapi_instrumentator ----------
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        logger.info("Prometheus HTTP instrumentation enabled for %s", service_name)
    except ImportError:
        logger.warning(
            "prometheus_fastapi_instrumentator not installed — /metrics endpoint disabled"
        )

    # --- Custom metrics via prometheus_client --------------------------------
    try:
        from prometheus_client import Counter, Gauge, Histogram

        metrics.inference_latency = _get_or_create(
            "onco_inference_latency_seconds",
            lambda: Histogram(
                "onco_inference_latency_seconds",
                "Inference latency in seconds",
                labelnames=["service", "patient_id_hash"],
            ),
        )

        metrics.active_patients = _get_or_create(
            "onco_active_patients_total",
            lambda: Gauge(
                "onco_active_patients_total",
                "Number of active patients",
                labelnames=["service", "ward_id"],
            ),
        )

        metrics.alert_generated = _get_or_create(
            "onco_alert_generated_total",
            lambda: Counter(
                "onco_alert_generated_total",
                "Total alerts generated",
                labelnames=["service", "alert_type", "priority"],
            ),
        )

        metrics.score_staleness = _get_or_create(
            "onco_score_staleness_total",
            lambda: Counter(
                "onco_score_staleness_total",
                "Total stale score events",
                labelnames=["service"],
            ),
        )

        metrics.redis_stream_lag = _get_or_create(
            "onco_redis_stream_lag",
            lambda: Gauge(
                "onco_redis_stream_lag",
                "Redis stream consumer lag",
                labelnames=["service", "stream_name"],
            ),
        )

        logger.info("Custom Prometheus metrics registered for %s", service_name)

    except ImportError:
        logger.warning(
            "prometheus_client not installed — custom metrics disabled"
        )

    return metrics


# ---------------------------------------------------------------------------
# Error rate monitor (subtask 23.4)
# ---------------------------------------------------------------------------

# Rolling window storage: (timestamp, is_error)
_error_window: deque[tuple[float, bool]] = deque()
_WINDOW_SECONDS = 300  # 5 minutes
_error_monitor_task: asyncio.Task | None = None  # type: ignore[type-arg]


async def start_error_rate_monitor(
    app: FastAPI,
    service_name: str,
    redis_url: str,
) -> asyncio.Task:  # type: ignore[type-arg]
    """
    Start a background task that publishes an operational alert to Redis
    if error rate > 5% (with > 10 requests) in a rolling 5-minute window.

    NOTE: The error-tracking middleware must be registered at app creation
    time via setup_metrics(). This function only starts the monitor task.

    Returns the background asyncio.Task so it can be cancelled on shutdown.
    """
    global _error_monitor_task
    async def _monitor_loop() -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.warning("redis not installed — error rate monitor disabled")
            return

        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            while True:
                await asyncio.sleep(60)
                _prune_window()

                total = len(_error_window)
                errors = sum(1 for _, is_err in _error_window if is_err)

                if total > 10:
                    error_rate = errors / total
                    if error_rate > 0.05:
                        payload = {
                            "alert_type": "operational",
                            "service": service_name,
                            "error_rate": round(error_rate, 4),
                            "window_seconds": _WINDOW_SECONDS,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        try:
                            import json
                            await redis_client.xadd(
                                "onco:alert:generated",
                                {"data": json.dumps(payload)},
                                maxlen=100_000,
                                approximate=True,
                            )
                            logger.warning(
                                "Operational alert published: service=%s error_rate=%.2f%%",
                                service_name,
                                error_rate * 100,
                            )
                        except Exception as exc:
                            logger.error("Failed to publish operational alert: %s", exc)
        finally:
            await redis_client.aclose()

    def _prune_window() -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - _WINDOW_SECONDS
        while _error_window and _error_window[0][0] < cutoff:
            _error_window.popleft()

    _error_monitor_task = asyncio.create_task(_monitor_loop(), name=f"{service_name}-error-monitor")
    logger.info("Error rate monitor started for %s", service_name)
    return _error_monitor_task


# ---------------------------------------------------------------------------
# Graceful shutdown helper (subtask 23.6)
# ---------------------------------------------------------------------------

async def graceful_shutdown(
    tasks: list[asyncio.Task],  # type: ignore[type-arg]
    timeout: float = 30.0,
) -> None:
    """
    Cancel all provided tasks and wait for them to finish within the timeout.
    Logs a warning if the timeout is exceeded.
    """
    if not tasks:
        return

    for task in tasks:
        if task and not task.done():
            task.cancel()

    try:
        await asyncio.wait_for(
            asyncio.gather(*[t for t in tasks if t], return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Graceful shutdown timed out after %.1f seconds — some tasks may not have finished",
            timeout,
        )
