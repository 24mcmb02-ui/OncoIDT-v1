"""
OncoIDT reusable /health and /ready FastAPI router.

Mount in any service:
    from shared.health import make_health_router
    app.include_router(make_health_router(service_name="ingestion-service", readiness_checks={"db": check_db}))
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.config import get_settings
from shared.schemas import HealthResponse, ReadyResponse

logger = logging.getLogger(__name__)

ReadinessCheck = Callable[[], Awaitable[bool]]


def make_health_router(
    service_name: str | None = None,
    readiness_checks: dict[str, ReadinessCheck] | None = None,
) -> APIRouter:
    """
    Factory that returns a router with /health and /ready endpoints.

    Args:
        service_name: Human-readable service identifier (defaults to settings.service_name).
        readiness_checks: Mapping of check_name → async callable returning bool.
                          All checks must pass for /ready to return 200.
    """
    router = APIRouter(tags=["observability"])
    settings = get_settings()
    _service_name = service_name or settings.service_name
    _checks: dict[str, ReadinessCheck] = readiness_checks or {}

    @router.get("/health", response_model=HealthResponse, summary="Liveness probe")
    async def health() -> HealthResponse:
        """Always returns 200 while the process is alive."""
        return HealthResponse(
            status="ok",
            service=_service_name,
            version=settings.service_version,
        )

    @router.get("/ready", summary="Readiness probe")
    async def ready() -> JSONResponse:
        """
        Runs all registered readiness checks.
        Returns 200 if all pass, 503 if any fail.
        """
        results: dict[str, bool] = {}
        all_ok = True

        for name, check_fn in _checks.items():
            try:
                ok = await check_fn()
            except Exception as exc:
                logger.warning("Readiness check '%s' raised: %s", name, exc)
                ok = False
            results[name] = ok
            if not ok:
                all_ok = False

        payload = ReadyResponse(ready=all_ok, checks=results)
        status_code = 200 if all_ok else 503
        return JSONResponse(content=payload.model_dump(), status_code=status_code)

    return router
