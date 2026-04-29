"""
OncoIDT JWT authentication and RBAC.

Usage in a FastAPI route:
    @router.get("/sensitive")
    async def sensitive(user: CurrentUser = Depends(require_role(Role.CLINICIAN))):
        ...
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from shared.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RBAC role enum
# ---------------------------------------------------------------------------

class Role(str, Enum):
    CLINICIAN = "Clinician"
    CHARGE_NURSE = "Charge_Nurse"
    INFECTION_CONTROL_OFFICER = "Infection_Control_Officer"
    RESEARCH_ANALYST = "Research_Analyst"
    SYSTEM_ADMINISTRATOR = "System_Administrator"
    AUDIT_REVIEWER = "Audit_Reviewer"


# Role hierarchy — higher index = more privilege for simple comparisons
ROLE_HIERARCHY: list[Role] = [
    Role.AUDIT_REVIEWER,
    Role.RESEARCH_ANALYST,
    Role.CLINICIAN,
    Role.CHARGE_NURSE,
    Role.INFECTION_CONTROL_OFFICER,
    Role.SYSTEM_ADMINISTRATOR,
]


# ---------------------------------------------------------------------------
# Token payload
# ---------------------------------------------------------------------------

class TokenPayload:
    def __init__(self, user_id: str, role: Role, ward_ids: list[str]) -> None:
        self.user_id = user_id
        self.role = role
        self.ward_ids = ward_ids

    def has_role(self, *roles: Role) -> bool:
        return self.role in roles

    def can_access_ward(self, ward_id: str) -> bool:
        if self.role in (Role.SYSTEM_ADMINISTRATOR, Role.INFECTION_CONTROL_OFFICER):
            return True
        return ward_id in self.ward_ids


# ---------------------------------------------------------------------------
# Bearer scheme
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=True)


def _decode_token(token: str) -> TokenPayload:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}")

    try:
        return TokenPayload(
            user_id=payload["sub"],
            role=Role(payload["role"]),
            ward_ids=payload.get("ward_ids", []),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Malformed token payload: {exc}")


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> TokenPayload:
    return _decode_token(credentials.credentials)


CurrentUser = Annotated[TokenPayload, Depends(get_current_user)]


# ---------------------------------------------------------------------------
# require_role dependency factory
# ---------------------------------------------------------------------------

def require_role(*allowed_roles: Role):
    """
    FastAPI dependency factory.  Returns a dependency that validates the
    authenticated user has one of the specified roles.

    Example:
        @router.delete("/admin/thing")
        async def delete_thing(user: Annotated[TokenPayload, Depends(require_role(Role.SYSTEM_ADMINISTRATOR))]):
            ...
    """
    async def _dependency(
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    ) -> TokenPayload:
        user = _decode_token(credentials.credentials)
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not authorised for this endpoint. Required: {[r.value for r in allowed_roles]}",
            )
        return user

    return _dependency


# ---------------------------------------------------------------------------
# Token creation helper (used by API gateway login endpoint)
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, role: Role, ward_ids: list[str]) -> str:
    import datetime

    settings = get_settings()
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": user_id,
        "role": role.value,
        "ward_ids": ward_ids,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=settings.jwt_expiry_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
