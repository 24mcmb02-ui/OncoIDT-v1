"""
OncoIDT Ingestion Service — FastAPI application.

Endpoints:
  POST /ingest/{source_type}   — ingest raw bytes via registered adapter
  POST /ingest/batch           — multipart CSV/JSON backfill upload
  GET  /health                 — liveness probe
  GET  /ready                  — readiness probe

Background tasks:
  - Source unavailability monitor: emits alert if a source is silent > 5 minutes
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Path, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import text

from shared.config import get_settings
from shared.db import close_engine, get_db_session, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, get_redis, publish_event
from services.ingestion.adapters.base import registry
from services.ingestion.adapters.csv_batch import CSVBatchAdapter
from services.ingestion.adapters.fhir import FHIRAdapter
from services.ingestion.adapters.json_push import JSONPushAdapter
from services.ingestion.dedup import DedupResult, deduplicate

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

# ---------------------------------------------------------------------------
# Register built-in adapters
# ---------------------------------------------------------------------------
registry.register(FHIRAdapter())
registry.register(CSVBatchAdapter())
registry.register(JSONPushAdapter())

# Try to register HL7v2 adapter (requires hl7apy)
try:
    from services.ingestion.adapters.hl7v2 import HL7v2Adapter
    registry.register(HL7v2Adapter())
except Exception:
    logger.warning("HL7v2 adapter not registered (hl7apy may not be installed)")

# Register synthetic adapter for demo/testing
try:
    from services.ingestion.adapters.synthetic import SyntheticAdapter
    registry.register(SyntheticAdapter())
except Exception:
    logger.warning("Synthetic adapter not registered")

# ---------------------------------------------------------------------------
# Source activity tracker for unavailability monitoring
# ---------------------------------------------------------------------------
_last_seen: dict[str, datetime] = {}
_SOURCE_SILENCE_SECONDS = 300  # 5 minutes


async def _source_unavailability_monitor() -> None:
    """Background task: emit alert if any source is silent > 5 minutes."""
    redis = get_redis()
    while True:
        await asyncio.sleep(60)
        now = datetime.now(timezone.utc)
        for source_type, last_ts in list(_last_seen.items()):
            silence_seconds = (now - last_ts).total_seconds()
            if silence_seconds > _SOURCE_SILENCE_SECONDS:
                logger.warning(
                    "Source '%s' has been silent for %.0f seconds — emitting alert",
                    source_type, silence_seconds,
                )
                try:
                    await publish_event(
                        "onco:alert:generated",
                        {
                            "alert_type": "source_unavailability",
                            "source_type": source_type,
                            "silence_seconds": silence_seconds,
                            "timestamp": now.isoformat(),
                        },
                        redis=redis,
                    )
                except Exception as exc:
                    logger.error("Failed to publish source_unavailability alert: %s", exc)


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
        redis = get_redis()
        await redis.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    monitor_task = asyncio.create_task(_source_unavailability_monitor())
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    logger.info("Ingestion service started. Registered adapters: %s", registry.registered_types)
    yield
    await graceful_shutdown([monitor_task, error_monitor_task])
    await close_engine()
    await close_redis_pool()
    logger.info("Ingestion service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Ingestion Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(
    make_health_router(
        service_name="ingestion-service",
        readiness_checks={"db": _check_db, "redis": _check_redis},
    )
)


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

async def _persist_record(session: Any, record: Any, dedup_hash: str) -> None:
    """Insert a CanonicalRecord into clinical_events."""
    import dataclasses

    payload_dict = dataclasses.asdict(record.payload)
    payload_dict["_dedup_hash"] = dedup_hash

    await session.execute(
        text(
            """
            INSERT INTO clinical_events
                (record_id, patient_id, record_type, timestamp_utc, ingested_at, payload, data_quality_flags)
            VALUES
                (:record_id, :patient_id, :record_type, :timestamp_utc, :ingested_at, :payload, :flags)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "record_id": record.record_id,
            "patient_id": record.patient_id,
            "record_type": record.record_type,
            "timestamp_utc": record.timestamp_utc,
            "ingested_at": record.ingested_at,
            "payload": json.dumps(payload_dict, default=str),
            "flags": json.dumps(
                [
                    {
                        "flag_type": f.flag_type,
                        "field_name": f.field_name,
                        "message": f.message,
                        "severity": f.severity,
                    }
                    for f in record.data_quality_flags
                ]
            ),
        },
    )


# ---------------------------------------------------------------------------
# POST /ingest/{source_type}
# ---------------------------------------------------------------------------

@app.post("/ingest/{source_type}", status_code=202)
async def ingest(
    source_type: Annotated[str, Path(description="Registered adapter source type")],
    request: Request,
) -> JSONResponse:
    """
    Accept raw bytes, route to the registered adapter, validate, deduplicate,
    persist to clinical_events, and publish to the Redis Stream.
    """
    try:
        adapter = registry.get_adapter(source_type)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No adapter registered for source_type '{source_type}'. "
                   f"Available: {registry.registered_types}",
        )

    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=422, detail="Request body is empty")

    # Parse
    try:
        records = adapter.parse(raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Validate temporal fields
    for record in records:
        if record.timestamp_utc.tzinfo is None:
            raise HTTPException(
                status_code=422,
                detail=f"Record {record.record_id}: timestamp_utc must be timezone-aware (UTC ISO 8601)",
            )

    # Update source activity tracker
    _last_seen[source_type] = datetime.now(timezone.utc)

    # Persist + publish
    accepted = 0
    duplicates = 0
    redis = get_redis()
    factory = get_session_factory()

    async with factory() as session:
        for record in records:
            result, dedup_hash = await deduplicate(session, record)
            if result == DedupResult.EXACT_DUPLICATE:
                duplicates += 1
                continue

            await _persist_record(session, record, dedup_hash)
            accepted += 1

            # Publish to Redis Stream
            try:
                import dataclasses
                await publish_event(
                    "onco:ingestion:patient_event",
                    {
                        "record_id": record.record_id,
                        "patient_id": record.patient_id,
                        "record_type": record.record_type,
                        "source_system": record.source_system,
                        "timestamp_utc": record.timestamp_utc.isoformat(),
                    },
                    redis=redis,
                )
            except Exception as exc:
                logger.error("Failed to publish event for record %s: %s", record.record_id, exc)

        await session.commit()

    return JSONResponse(
        status_code=202,
        content={
            "accepted": accepted,
            "duplicates_discarded": duplicates,
            "total_parsed": len(records),
        },
    )


# ---------------------------------------------------------------------------
# POST /ingest/synthetic/cohort  — bulk synthetic patient generation
# ---------------------------------------------------------------------------

@app.post("/ingest/synthetic/cohort", status_code=202)
async def ingest_synthetic_cohort(request: Request) -> JSONResponse:
    """
    Generate and ingest a synthetic oncology cohort.
    Body: {"num_patients": 50, "seed": 42}
    """
    body = await request.json()
    num_patients = int(body.get("num_patients", 50))
    seed = int(body.get("seed", 42))

    try:
        from services.ingestion.adapters.synthetic_cohort import (
            SyntheticCohortConfig, generate_cohort, EventRates
        )
        config = SyntheticCohortConfig(
            n_patients=num_patients,
            seed=seed,
            event_rates=EventRates(
                infection_per_admission=0.18,
                neutropenic_fever_per_admission=0.22,
                deterioration_per_admission=0.10,
            ),
        )
        cohort = generate_cohort(config)
        records = cohort.records
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cohort generation failed: {exc}") from exc

    accepted = 0
    duplicates = 0
    redis = get_redis()
    factory = get_session_factory()

    async with factory() as session:
        for record in records:
            try:
                result, dedup_hash = await deduplicate(session, record)
                if result == DedupResult.EXACT_DUPLICATE:
                    duplicates += 1
                    continue
                await _persist_record(session, record, dedup_hash)
                accepted += 1
                await publish_event(
                    "onco:ingestion:patient_event",
                    {
                        "record_id": record.record_id,
                        "patient_id": record.patient_id,
                        "record_type": record.record_type,
                        "source_system": record.source_system,
                        "timestamp_utc": record.timestamp_utc.isoformat(),
                    },
                    redis=redis,
                )
            except Exception as exc:
                logger.error("Failed to persist record %s: %s", record.record_id, exc)
        await session.commit()

    return JSONResponse(
        status_code=202,
        content={
            "num_patients": num_patients,
            "accepted": accepted,
            "duplicates_discarded": duplicates,
            "total_records": len(records),
        },
    )


# ---------------------------------------------------------------------------
# POST /ingest/batch
# ---------------------------------------------------------------------------

@app.post("/ingest/batch", status_code=202)
async def ingest_batch(
    file: Annotated[UploadFile, File(description="CSV or JSON file for backfill ingestion")],
) -> JSONResponse:
    """
    Accept a multipart file upload (CSV or JSON) and enqueue it to the
    low-priority backfill queue via Redis Stream.
    """
    if file.filename is None:
        raise HTTPException(status_code=422, detail="No filename provided")

    content_type = file.content_type or ""
    filename = file.filename.lower()

    if filename.endswith(".csv") or "csv" in content_type:
        source_type = "csv"
    elif filename.endswith(".json") or "json" in content_type:
        source_type = "json"
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Provide a .csv or .json file.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    job_id = str(uuid.uuid4())
    redis = get_redis()

    try:
        await publish_event(
            "onco:ingestion:backfill_job",
            {
                "job_id": job_id,
                "source_type": source_type,
                "filename": file.filename,
                "payload_size_bytes": len(raw),
                "payload": raw.decode("utf-8", errors="replace"),
                "enqueued_at": datetime.now(timezone.utc).isoformat(),
            },
            redis=redis,
        )
    except Exception as exc:
        logger.error("Failed to enqueue backfill job: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue backfill job") from exc

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "source_type": source_type, "status": "queued"},
    )
