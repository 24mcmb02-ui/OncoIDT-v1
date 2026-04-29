"""
OncoIDT API Gateway — Request logging middleware.

Writes every request to the audit log via shared/audit.py:
  - client identity (user_id or "anonymous")
  - endpoint (method + path)
  - timestamp
  - response status code

Requirements: 13.5
"""
from __future__ import annotations

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from shared.audit import append_audit_entry
from shared.db import get_session_factory

logger = logging.getLogger(__name__)

# Paths to skip audit logging (health/readiness probes are noisy)
_SKIP_AUDIT_PATHS = {"/health", "/ready", "/metrics"}


class RequestAuditMiddleware(BaseHTTPMiddleware):
    """
    Writes every API request to the audit_log table after the response is sent.
    Uses a fire-and-forget pattern to avoid adding latency to the request path.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        path = request.url.path
        if path not in _SKIP_AUDIT_PATHS:
            user = getattr(request.state, "user", None)
            user_id = user.user_id if user else "anonymous"

            # Fire-and-forget audit write
            import asyncio
            asyncio.create_task(
                _write_audit(
                    user_id=user_id,
                    method=request.method,
                    path=path,
                    status_code=response.status_code,
                    elapsed_ms=elapsed_ms,
                    query_string=str(request.url.query) if request.url.query else None,
                )
            )

        return response


async def _write_audit(
    user_id: str,
    method: str,
    path: str,
    status_code: int,
    elapsed_ms: int,
    query_string: str | None,
) -> None:
    """Write a single audit entry for an API request."""
    try:
        factory = get_session_factory()
        async with factory() as session:
            await append_audit_entry(
                session,
                user_id=user_id,
                action=f"api_request:{method}",
                resource_type="api_endpoint",
                resource_id=path,
                details={
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "elapsed_ms": elapsed_ms,
                    "query_string": query_string,
                },
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to write request audit entry for %s %s: %s", method, path, exc)
