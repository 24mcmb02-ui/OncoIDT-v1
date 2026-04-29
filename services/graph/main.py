"""
OncoIDT Graph Service — FastAPI application.

Exposes:
  POST /graph/events        — consume Redis streams (background task)
  GET  /graph/neighborhood/{node_id}  — k-hop query
  GET  /graph/contact-trace/{patient_id} — contact tracing
  POST /graph/snapshots     — trigger manual snapshot

Requirements: 7.3, 7.6, 7.7, 2.4
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.graph.graph_ops import (
    close_neo4j_driver,
    get_neo4j_driver,
    record_pathogen_exposure,
    upsert_bed_assignment,
    upsert_colocation,
    upsert_patient_node,
    upsert_staff_contact,
)
from services.graph.queries import (
    get_contact_trace,
    get_infection_signal_neighbors,
    get_k_hop_neighborhood,
)
from services.graph.snapshots import take_snapshot
from shared.config import get_settings
from shared.db import close_engine, get_db_session, get_session_factory
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor
from shared.redis_client import close_redis_pool, consume_events, get_redis

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------

async def _check_neo4j() -> bool:
    try:
        driver = get_neo4j_driver()
        await driver.verify_connectivity()
        return True
    except Exception:
        return False


async def _check_db() -> bool:
    try:
        from sqlalchemy import text
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        r = get_redis()
        await r.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stream consumer background task
# ---------------------------------------------------------------------------

_consumer_task: asyncio.Task | None = None  # type: ignore[type-arg]


async def _consume_streams() -> None:
    """
    Consume onco:ingestion:patient_event and onco:ward:state_change streams
    and dispatch to graph CRUD operations.
    Requirement 7.3
    """
    streams = [
        ("onco:ingestion:patient_event", "graph-service", "graph-consumer-1"),
        ("onco:ward:state_change", "graph-service-ward", "graph-consumer-ward-1"),
    ]

    async def _handle_patient_event(data: dict[str, Any]) -> None:
        record_type = data.get("record_type", "")
        patient_id = data.get("patient_id")
        if not patient_id:
            return

        ward_id = data.get("ward_id", "unknown")
        payload = data.get("payload", {})

        # Always upsert the patient node
        await upsert_patient_node(
            patient_id=patient_id,
            ward_id=ward_id,
            bed_id=data.get("bed_id"),
        )

        if record_type == "event":
            event_type = payload.get("event_type", "")
            if event_type in ("admission", "transfer"):
                bed_id = payload.get("metadata", {}).get("bed_id")
                if bed_id:
                    ts_str = data.get("timestamp_utc")
                    ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
                    await upsert_bed_assignment(patient_id, bed_id, ts)

            elif event_type == "infection_confirmed":
                pathogen = payload.get("metadata", {}).get("pathogen", "unknown")
                await record_pathogen_exposure(
                    patient_id=patient_id,
                    pathogen=pathogen,
                    confidence=payload.get("metadata", {}).get("confidence", 1.0),
                    source_event_id=data.get("record_id", ""),
                )

    async def _handle_ward_event(data: dict[str, Any]) -> None:
        event_type = data.get("event_type", "")
        if event_type == "staff_contact":
            await upsert_staff_contact(
                patient_id=data["patient_id"],
                staff_id=data["staff_id"],
                role=data.get("role", "unknown"),
                shift_start=datetime.fromisoformat(data["shift_start"]),
                shift_end=(
                    datetime.fromisoformat(data["shift_end"])
                    if data.get("shift_end") else None
                ),
            )
        elif event_type == "colocation":
            await upsert_colocation(
                patient_a=data["patient_a"],
                patient_b=data["patient_b"],
                room_id=data["room_id"],
                overlap_start=datetime.fromisoformat(data["overlap_start"]),
                overlap_end=(
                    datetime.fromisoformat(data["overlap_end"])
                    if data.get("overlap_end") else None
                ),
            )

    async def _run_consumer(stream: str, group: str, consumer: str) -> None:
        async for msg_id, data in consume_events(stream, group, consumer):
            try:
                if stream == "onco:ingestion:patient_event":
                    await _handle_patient_event(data)
                else:
                    await _handle_ward_event(data)
            except Exception as exc:
                logger.error("Error processing %s message %s: %s", stream, msg_id, exc)

    await asyncio.gather(*[_run_consumer(s, g, c) for s, g, c in streams])


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _consumer_task
    logger.info("Graph service starting up")
    _consumer_task = asyncio.create_task(_consume_streams())
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    yield
    logger.info("Graph service shutting down")
    await graceful_shutdown([_consumer_task, error_monitor_task])
    await close_neo4j_driver()
    await close_engine()
    await close_redis_pool()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Graph Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)
app.include_router(
    make_health_router(
        service_name="graph-service",
        readiness_checks={
            "neo4j": _check_neo4j,
            "db": _check_db,
            "redis": _check_redis,
        },
    )
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class NeighborhoodResponse(BaseModel):
    node_id: str
    neighbors: list[dict[str, Any]]


class ContactTraceResponse(BaseModel):
    patient_id: str
    since: str
    contacts: list[dict[str, Any]]


class SnapshotRequest(BaseModel):
    ward_id: str
    snapshot_timestamp: datetime | None = None


class SnapshotResponse(BaseModel):
    snapshot_id: str
    ward_id: str
    snapshot_timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/graph/events", summary="Trigger stream consumption (no-op — runs as background task)")
async def post_graph_events() -> JSONResponse:
    """
    The stream consumer runs as a background task started at lifespan.
    This endpoint exists for manual triggering / health verification.
    """
    running = _consumer_task is not None and not _consumer_task.done()
    return JSONResponse({"consumer_running": running})


@app.get(
    "/graph/neighborhood/{node_id}",
    response_model=NeighborhoodResponse,
    summary="k-hop neighborhood query",
)
async def get_neighborhood(
    node_id: str,
    k: int = Query(default=2, ge=1, le=3, description="Maximum hop distance"),
    edge_types: list[str] | None = Query(default=None, description="Edge type filter"),
    since: datetime | None = Query(default=None, description="Minimum edge timestamp (ISO 8601)"),
) -> NeighborhoodResponse:
    """
    Return all nodes within k hops of the given node, with optional
    edge-type and time filters.
    Requirement 7.6
    """
    try:
        neighbors = await get_k_hop_neighborhood(
            node_id=node_id,
            k=k,
            edge_types=edge_types,
            since=since,
        )
    except Exception as exc:
        logger.error("Neighborhood query failed for %s: %s", node_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return NeighborhoodResponse(node_id=node_id, neighbors=neighbors)


@app.get(
    "/graph/contact-trace/{patient_id}",
    response_model=ContactTraceResponse,
    summary="Contact tracing for a patient",
)
async def contact_trace(
    patient_id: str,
    since: datetime = Query(..., description="Start of contact trace window (ISO 8601)"),
) -> ContactTraceResponse:
    """
    Return the CO_LOCATED + TREATED_BY contact chain for a patient since
    the given timestamp.
    Requirement 7.6
    """
    try:
        contacts = await get_contact_trace(patient_id=patient_id, since=since)
    except Exception as exc:
        logger.error("Contact trace failed for %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ContactTraceResponse(
        patient_id=patient_id,
        since=since.isoformat(),
        contacts=contacts,
    )


@app.post(
    "/graph/snapshots",
    response_model=SnapshotResponse,
    summary="Trigger a manual graph snapshot",
)
async def create_snapshot(body: SnapshotRequest) -> SnapshotResponse:
    """
    Serialise the current graph state for the given ward and persist it
    to the graph_snapshots PostgreSQL table.
    Requirement 7.4
    """
    factory = get_session_factory()
    async with factory() as db:
        try:
            snapshot_id = await take_snapshot(
                ward_id=body.ward_id,
                db=db,
                snapshot_timestamp=body.snapshot_timestamp,
            )
        except Exception as exc:
            logger.error("Snapshot failed for ward %s: %s", body.ward_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    ts = body.snapshot_timestamp or datetime.utcnow()
    return SnapshotResponse(
        snapshot_id=snapshot_id,
        ward_id=body.ward_id,
        snapshot_timestamp=ts.isoformat(),
    )
