"""
OncoIDT API Gateway — JWT validation and RBAC middleware.

JWTValidationMiddleware:
  - Validates Bearer token on every request (except /health, /ready, /auth/login, /docs, /openapi.json)
  - Returns HTTP 401 on missing or invalid token

RBACMiddleware:
  - Enforces ward-scoped access for ward-specific endpoints
  - Returns HTTP 403 and writes audit log entry on unauthorized access

Login endpoint:
  POST /auth/login  — issues JWT with user_id, role, ward_ids

Requirements: 17.3, 13.3, 17.7
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from shared.auth import Role, TokenPayload, create_access_token, _decode_token
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Paths that bypass JWT validation
_PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/auth/login",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/v1/docs",
    "/api/v1/openapi.json",
}


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith("/api/v1/docs")


# ---------------------------------------------------------------------------
# JWT Validation Middleware
# ---------------------------------------------------------------------------

class JWTValidationMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that validates the Bearer JWT on every non-public request.
    Attaches the decoded TokenPayload to request.state.user on success.
    Returns HTTP 401 on missing or invalid token.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _is_public(request.url.path):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or malformed Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.removeprefix("Bearer ").strip()
        try:
            user: TokenPayload = _decode_token(token)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.user = user
        return await call_next(request)


# ---------------------------------------------------------------------------
# RBAC Middleware
# ---------------------------------------------------------------------------

class RBACMiddleware(BaseHTTPMiddleware):
    """
    Enforces ward-scoped access for endpoints under /api/v1/ward/{ward_id}.
    Writes an audit log entry on unauthorized access and returns HTTP 403.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _is_public(request.url.path):
            return await call_next(request)

        user: TokenPayload | None = getattr(request.state, "user", None)
        if user is None:
            # JWT middleware will have already rejected this; pass through
            return await call_next(request)

        # Ward-scoped access check for /api/v1/ward/{ward_id}
        path_parts = request.url.path.split("/")
        if "ward" in path_parts:
            try:
                ward_idx = path_parts.index("ward")
                ward_id = path_parts[ward_idx + 1] if ward_idx + 1 < len(path_parts) else None
            except (ValueError, IndexError):
                ward_id = None

            if ward_id and not user.can_access_ward(ward_id):
                # Write audit log entry asynchronously (best-effort)
                await _write_forbidden_audit(request, user, ward_id)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": f"Access to ward '{ward_id}' is not permitted for this account"
                    },
                )

        return await call_next(request)


async def _write_forbidden_audit(request: Request, user: TokenPayload, ward_id: str) -> None:
    """Best-effort audit log write for forbidden access attempts."""
    try:
        from shared.audit import append_audit_entry
        from shared.db import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            await append_audit_entry(
                session,
                user_id=user.user_id,
                action="forbidden_access_attempt",
                resource_type="ward",
                resource_id=ward_id,
                details={
                    "path": request.url.path,
                    "method": request.method,
                    "role": user.role.value,
                    "authorized_wards": user.ward_ids,
                },
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to write forbidden audit entry: %s", exc)


# ---------------------------------------------------------------------------
# Login router
# ---------------------------------------------------------------------------

login_router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    user_id: str
    password: str
    role: Role
    ward_ids: list[str] = []


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


@login_router.post("/auth/login", response_model=LoginResponse, summary="Issue JWT access token")
async def login(body: LoginRequest) -> LoginResponse:
    """
    Issue a JWT access token.

    In production this endpoint must validate credentials against an identity
    provider. This implementation issues a token directly for development/research
    use — replace the credential check with your IdP integration.
    """
    # TODO: replace with real credential validation against user store / IdP
    token = create_access_token(
        user_id=body.user_id,
        role=body.role,
        ward_ids=body.ward_ids,
    )
    return LoginResponse(
        access_token=token,
        expires_in_minutes=settings.jwt_expiry_minutes,
    )
