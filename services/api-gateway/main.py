"""
OncoIDT API Gateway — FastAPI Application.

Middleware stack (applied in order):
  1. JWTValidationMiddleware  — validate Bearer token, attach user to request.state
  2. RBACMiddleware           — enforce ward-scoped access, write forbidden audit entries
  3. RateLimitMiddleware      — Redis sliding window 1000 req/min per client
  4. RequestAuditMiddleware   — write every request to audit_log

Routers:
  /auth/login                — issue JWT
  /api/v1/patients           — patient twin CRUD
  /api/v1/ward               — ward twin + heatmap
  /api/v1/simulations        — what-if simulation (proxy)
  /api/v1/alerts             — alert management (proxy)
  /api/v1/admin              — rule management + model promotion
  /api/v1/research           — de-identified research export
  /api/v1/fhir               — FHIR R4 endpoints
  /health, /ready            — liveness + readiness probes

Requirements: 13.1–13.8, 17.3, 17.6, 17.7
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

try:
    # Absolute imports (Docker: copied as services.api_gateway)
    from services.api_gateway.middleware.auth import JWTValidationMiddleware, RBACMiddleware, login_router
    from services.api_gateway.middleware.logging import RequestAuditMiddleware
    from services.api_gateway.middleware.rate_limit import RateLimitMiddleware
    from services.api_gateway.routers.admin import router as admin_router
    from services.api_gateway.routers.alerts import router as alerts_router
    from services.api_gateway.routers.fhir import router as fhir_router
    from services.api_gateway.routers.patients import router as patients_router
    from services.api_gateway.routers.research import router as research_router
    from services.api_gateway.routers.simulations import router as simulations_router
    from services.api_gateway.routers.ward import router as ward_router
except ImportError:
    # Relative imports (local dev: services/api-gateway/)
    from .middleware.auth import JWTValidationMiddleware, RBACMiddleware, login_router  # type: ignore[no-redef]
    from .middleware.logging import RequestAuditMiddleware  # type: ignore[no-redef]
    from .middleware.rate_limit import RateLimitMiddleware  # type: ignore[no-redef]
    from .routers.admin import router as admin_router  # type: ignore[no-redef]
    from .routers.alerts import router as alerts_router  # type: ignore[no-redef]
    from .routers.fhir import router as fhir_router  # type: ignore[no-redef]
    from .routers.patients import router as patients_router  # type: ignore[no-redef]
    from .routers.research import router as research_router  # type: ignore[no-redef]
    from .routers.simulations import router as simulations_router  # type: ignore[no-redef]
    from .routers.ward import router as ward_router  # type: ignore[no-redef]
from shared.config import get_settings
from shared.db import close_engine, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, get_redis

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------

async def _check_db() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        await get_redis().ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    logger.info("API Gateway started.")
    yield
    await graceful_shutdown([error_monitor_task])
    await close_engine()
    await close_redis_pool()
    logger.info("API Gateway shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT API Gateway",
    version=settings.service_version,
    description="Versioned REST + FHIR R4 API for the OncoIDT platform",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

# ---------------------------------------------------------------------------
# Middleware (added in reverse order — last added = outermost)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit logging (outermost — captures final status code)
app.add_middleware(RequestAuditMiddleware)

# Rate limiting
app.add_middleware(RateLimitMiddleware)

# RBAC ward-scope enforcement
app.add_middleware(RBACMiddleware)

# JWT validation (innermost — runs first, attaches user to request.state)
app.add_middleware(JWTValidationMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(
    make_health_router(
        service_name="api-gateway",
        readiness_checks={"db": _check_db, "redis": _check_redis},
    )
)

app.include_router(login_router)
app.include_router(patients_router)
app.include_router(ward_router)
app.include_router(simulations_router)
app.include_router(alerts_router)
app.include_router(admin_router)
app.include_router(research_router)
app.include_router(fhir_router)
