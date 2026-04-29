"""
OncoIDT Feature Store Service — FastAPI application.

Endpoints:
  GET  /features/{patient_id}?as_of=<iso8601>&version=<str>
       Single-patient low-latency feature retrieval (< 100ms SLA).

  POST /features/batch
       Batch retrieval for training dataset construction.
       Body: list of {"patient_id": str, "timestamp": iso8601}

  POST /features/backfill
       Trigger backfill job for a new feature version across the cohort.
       Body: {"feature_version": str, "patient_ids": [str, ...] | null}

  GET  /health  — liveness probe
  GET  /ready   — readiness probe
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from shared.config import get_settings
from shared.db import close_engine, get_db_session, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from services.feature_store.features import compute_all_features

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class BatchItem(BaseModel):
    patient_id: str
    timestamp: datetime


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=10_000)


class BackfillRequest(BaseModel):
    feature_version: str
    patient_ids: list[str] | None = None  # None = all active patients


class FeatureResponse(BaseModel):
    patient_id: str
    as_of: datetime
    feature_version: str
    snapshot_id: str
    features: dict[str, Any]


class BatchResponseItem(BaseModel):
    patient_id: str
    timestamp: datetime
    feature_version: str
    snapshot_id: str
    features: dict[str, Any]
    error: str | None = None


class BackfillResponse(BaseModel):
    job_id: str
    feature_version: str
    patient_count: int
    status: str = "queued"


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


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _ensure_lineage(session: Any, feature_version: str) -> str:
    """
    Return the lineage_id for a feature version, creating a new record
    if this is the first use of that version (Requirement 19.6).
    """
    row = await session.execute(
        text(
            "SELECT lineage_id FROM feature_lineage WHERE feature_version = :v AND modality = 'structured'"
        ),
        {"v": feature_version},
    )
    existing = row.fetchone()
    if existing:
        return str(existing[0])

    lineage_id = str(uuid.uuid4())
    await session.execute(
        text(
            """
            INSERT INTO feature_lineage
                (lineage_id, feature_version, modality, source_fields,
                 transformation_code_hash, deployed_at, description)
            VALUES
                (:lineage_id, :version, 'structured', :source_fields,
                 :code_hash, :deployed_at, :description)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "lineage_id": lineage_id,
            "version": feature_version,
            "source_fields": json.dumps(
                ["clinical_events.vital", "clinical_events.lab",
                 "clinical_events.medication", "clinical_events.event"]
            ),
            "code_hash": _compute_code_hash(),
            "deployed_at": datetime.now(timezone.utc),
            "description": f"Auto-registered feature version {feature_version}",
        },
    )
    logger.info("Registered new feature lineage: version=%s lineage_id=%s", feature_version, lineage_id)
    return lineage_id


def _compute_code_hash() -> str:
    """SHA-256 of the features module source for lineage tracking."""
    import hashlib
    import inspect
    from services.feature_store import features as feat_module

    src = inspect.getsource(feat_module)
    return hashlib.sha256(src.encode()).hexdigest()


async def _persist_snapshot(
    session: Any,
    patient_id: str,
    as_of: datetime,
    feature_version: str,
    features: dict[str, Any],
    lineage_id: str,
) -> str:
    """Upsert a feature snapshot into feature_store. Returns snapshot_id."""
    snapshot_id = str(uuid.uuid4())
    await session.execute(
        text(
            """
            INSERT INTO feature_store
                (patient_id, timestamp, feature_version, modality, features, lineage_id)
            VALUES
                (:patient_id, :ts, :version, 'structured', :features, :lineage_id)
            ON CONFLICT (patient_id, timestamp, feature_version, modality)
            DO UPDATE SET features = EXCLUDED.features, lineage_id = EXCLUDED.lineage_id
            """
        ),
        {
            "patient_id": patient_id,
            "ts": as_of,
            "version": feature_version,
            "features": json.dumps(features, default=str),
            "lineage_id": lineage_id,
        },
    )
    return snapshot_id


async def _get_cached_snapshot(
    session: Any,
    patient_id: str,
    as_of: datetime,
    feature_version: str,
) -> dict[str, Any] | None:
    """Return a cached feature snapshot if one exists for this exact key."""
    row = await session.execute(
        text(
            """
            SELECT features
            FROM feature_store
            WHERE patient_id = :patient_id
              AND timestamp = :ts
              AND feature_version = :version
              AND modality = 'structured'
            LIMIT 1
            """
        ),
        {"patient_id": patient_id, "ts": as_of, "version": feature_version},
    )
    result = row.fetchone()
    if result:
        raw = result[0]
        return raw if isinstance(raw, dict) else json.loads(raw)
    return None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    logger.info("Feature store service started.")
    yield
    await graceful_shutdown([error_monitor_task])
    await close_engine()
    logger.info("Feature store service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Feature Store Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="feature-store-service",
        readiness_checks={"db": _check_db},
    )
)


# ---------------------------------------------------------------------------
# GET /features/{patient_id}
# ---------------------------------------------------------------------------


@app.get(
    "/features/{patient_id}",
    response_model=FeatureResponse,
    summary="Single-patient feature retrieval (< 100ms SLA)",
)
async def get_features(
    patient_id: str,
    as_of: Annotated[
        datetime | None,
        Query(description="Point-in-time timestamp (ISO 8601). Defaults to now."),
    ] = None,
    version: Annotated[
        str,
        Query(description="Feature version string"),
    ] = settings.feature_store_default_version,
) -> FeatureResponse:
    """
    Retrieve (or compute and cache) the feature vector for a patient at
    the given point-in-time timestamp.  Requirement 19.1, 19.4.
    """
    effective_as_of = as_of or datetime.now(timezone.utc)
    if effective_as_of.tzinfo is None:
        effective_as_of = effective_as_of.replace(tzinfo=timezone.utc)

    factory = get_session_factory()
    async with factory() as session:
        # Check cache first
        cached = await _get_cached_snapshot(session, patient_id, effective_as_of, version)
        if cached:
            return FeatureResponse(
                patient_id=patient_id,
                as_of=effective_as_of,
                feature_version=version,
                snapshot_id=cached.get("_snapshot_id", str(uuid.uuid4())),
                features=cached,
            )

        # Compute features
        try:
            features = await compute_all_features(patient_id, effective_as_of, session)
        except Exception as exc:
            logger.error(
                "Feature computation failed for patient=%s as_of=%s: %s",
                patient_id, effective_as_of, exc,
            )
            raise HTTPException(status_code=500, detail=f"Feature computation error: {exc}") from exc

        # Persist snapshot
        lineage_id = await _ensure_lineage(session, version)
        snapshot_id = await _persist_snapshot(
            session, patient_id, effective_as_of, version, features, lineage_id
        )
        features["_snapshot_id"] = snapshot_id
        await session.commit()

    return FeatureResponse(
        patient_id=patient_id,
        as_of=effective_as_of,
        feature_version=version,
        snapshot_id=snapshot_id,
        features=features,
    )


# ---------------------------------------------------------------------------
# POST /features/batch
# ---------------------------------------------------------------------------


@app.post(
    "/features/batch",
    summary="Batch feature retrieval for training dataset construction",
)
async def get_features_batch(body: BatchRequest) -> JSONResponse:
    """
    Retrieve feature vectors for a list of (patient_id, timestamp) pairs.
    Returns one feature vector per item.  Requirement 19.4, 19.5.
    """
    factory = get_session_factory()
    results: list[dict[str, Any]] = []

    async with factory() as session:
        lineage_cache: dict[str, str] = {}

        for item in body.items:
            as_of = item.timestamp
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)

            try:
                # Check cache
                cached = await _get_cached_snapshot(
                    session, item.patient_id, as_of,
                    settings.feature_store_default_version,
                )
                if cached:
                    results.append(
                        BatchResponseItem(
                            patient_id=item.patient_id,
                            timestamp=as_of,
                            feature_version=settings.feature_store_default_version,
                            snapshot_id=cached.get("_snapshot_id", str(uuid.uuid4())),
                            features=cached,
                        ).model_dump(mode="json")
                    )
                    continue

                features = await compute_all_features(item.patient_id, as_of, session)
                version = settings.feature_store_default_version

                if version not in lineage_cache:
                    lineage_cache[version] = await _ensure_lineage(session, version)

                snapshot_id = await _persist_snapshot(
                    session, item.patient_id, as_of, version,
                    features, lineage_cache[version],
                )
                features["_snapshot_id"] = snapshot_id

                results.append(
                    BatchResponseItem(
                        patient_id=item.patient_id,
                        timestamp=as_of,
                        feature_version=version,
                        snapshot_id=snapshot_id,
                        features=features,
                    ).model_dump(mode="json")
                )

            except Exception as exc:
                logger.error(
                    "Batch feature error patient=%s ts=%s: %s",
                    item.patient_id, as_of, exc,
                )
                results.append(
                    BatchResponseItem(
                        patient_id=item.patient_id,
                        timestamp=as_of,
                        feature_version=settings.feature_store_default_version,
                        snapshot_id="",
                        features={},
                        error=str(exc),
                    ).model_dump(mode="json")
                )

        await session.commit()

    return JSONResponse(content={"results": results, "count": len(results)})


# ---------------------------------------------------------------------------
# POST /features/backfill
# ---------------------------------------------------------------------------


@app.post(
    "/features/backfill",
    response_model=BackfillResponse,
    summary="Trigger backfill job for a new feature version",
)
async def trigger_backfill(body: BackfillRequest) -> BackfillResponse:
    """
    Enqueue a background backfill job that recomputes features for all
    patients in the cohort (or a specified subset) under a new feature version.
    Requirement 19.5.
    """
    job_id = str(uuid.uuid4())

    # Resolve patient list
    factory = get_session_factory()
    async with factory() as session:
        if body.patient_ids:
            patient_ids = body.patient_ids
        else:
            # All patients with at least one clinical event
            result = await session.execute(
                text("SELECT DISTINCT patient_id FROM clinical_events")
            )
            patient_ids = [str(row[0]) for row in result.fetchall()]

    patient_count = len(patient_ids)
    logger.info(
        "Backfill job %s queued: version=%s patients=%d",
        job_id, body.feature_version, patient_count,
    )

    # Run backfill as a background task (fire-and-forget)
    asyncio.create_task(
        _run_backfill(job_id, body.feature_version, patient_ids)
    )

    return BackfillResponse(
        job_id=job_id,
        feature_version=body.feature_version,
        patient_count=patient_count,
        status="queued",
    )


async def _run_backfill(
    job_id: str,
    feature_version: str,
    patient_ids: list[str],
) -> None:
    """
    Background task: compute and persist feature snapshots for all
    (patient_id, as_of=now) pairs under the given feature version.
    """
    logger.info("Backfill %s started: %d patients", job_id, len(patient_ids))
    as_of = datetime.now(timezone.utc)
    factory = get_session_factory()
    success = 0
    errors = 0

    async with factory() as session:
        lineage_id = await _ensure_lineage(session, feature_version)

        for patient_id in patient_ids:
            try:
                features = await compute_all_features(patient_id, as_of, session)
                snapshot_id = await _persist_snapshot(
                    session, patient_id, as_of, feature_version, features, lineage_id
                )
                features["_snapshot_id"] = snapshot_id
                success += 1
            except Exception as exc:
                logger.error(
                    "Backfill %s: error for patient=%s: %s", job_id, patient_id, exc
                )
                errors += 1

        await session.commit()

    logger.info(
        "Backfill %s complete: success=%d errors=%d", job_id, success, errors
    )
