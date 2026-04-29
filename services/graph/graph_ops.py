"""
OncoIDT Graph Engine — Neo4j CRUD operations.

Implements upsert operations for all node and edge types in the
patient-ward knowledge graph (Requirements 7.1, 7.2, 7.3, 7.7).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver, AsyncSession  # type: ignore[import]

from shared.config import get_settings
from shared.redis_client import publish_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Driver factory
# ---------------------------------------------------------------------------

_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    """Return (or create) the shared async Neo4j driver."""
    global _driver
    if _driver is None:
        from neo4j import AsyncGraphDatabase  # type: ignore[import]
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dt(ts: datetime | str | None) -> str | None:
    """Normalise a datetime to ISO 8601 string for Neo4j storage."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    return ts


# ---------------------------------------------------------------------------
# Node upserts
# ---------------------------------------------------------------------------

async def upsert_patient_node(
    patient_id: str,
    ward_id: str,
    bed_id: str | None = None,
    *,
    mrn: str | None = None,
    age_years: int | None = None,
    sex: str | None = None,
    primary_diagnosis_icd10: str | None = None,
    chemo_regimen: str | None = None,
    chemo_cycle_phase: str | None = None,
    immunosuppression_score: float | None = None,
    status: str = "active",
    infection_risk_score: float | None = None,
    deterioration_risk_score: float | None = None,
    last_updated: datetime | None = None,
    driver: AsyncDriver | None = None,
) -> None:
    """
    Create or update a Patient node.  All properties are merged so that
    existing values are preserved when optional kwargs are None.
    Requirement 7.1
    """
    drv = driver or get_neo4j_driver()
    props: dict[str, Any] = {
        "id": patient_id,
        "ward_id": ward_id,
        "status": status,
    }
    if bed_id is not None:
        props["bed_id"] = bed_id
    if mrn is not None:
        props["mrn"] = mrn
    if age_years is not None:
        props["age_years"] = age_years
    if sex is not None:
        props["sex"] = sex
    if primary_diagnosis_icd10 is not None:
        props["primary_diagnosis_icd10"] = primary_diagnosis_icd10
    if chemo_regimen is not None:
        props["chemo_regimen"] = chemo_regimen
    if chemo_cycle_phase is not None:
        props["chemo_cycle_phase"] = chemo_cycle_phase
    if immunosuppression_score is not None:
        props["immunosuppression_score"] = immunosuppression_score
    if infection_risk_score is not None:
        props["infection_risk_score"] = infection_risk_score
    if deterioration_risk_score is not None:
        props["deterioration_risk_score"] = deterioration_risk_score
    props["last_updated"] = _dt(last_updated or datetime.utcnow())

    async with drv.session() as session:
        await session.run(
            """
            MERGE (p:Patient {id: $id})
            SET p += $props
            """,
            id=patient_id,
            props=props,
        )
    logger.debug("Upserted Patient node %s", patient_id)


async def upsert_ward_node(
    ward_id: str,
    ward_name: str | None = None,
    *,
    driver: AsyncDriver | None = None,
) -> None:
    """Create or update a Ward node."""
    drv = driver or get_neo4j_driver()
    props: dict[str, Any] = {"id": ward_id}
    if ward_name:
        props["name"] = ward_name
    async with drv.session() as session:
        await session.run(
            "MERGE (w:Ward {id: $id}) SET w += $props",
            id=ward_id,
            props=props,
        )


async def upsert_bed_node(
    bed_id: str,
    room_id: str,
    ward_id: str,
    zone: str | None = None,
    *,
    driver: AsyncDriver | None = None,
) -> None:
    """Create or update a Bed node and link it to its Room and Ward."""
    drv = driver or get_neo4j_driver()
    props: dict[str, Any] = {"id": bed_id, "room_id": room_id, "ward_id": ward_id}
    if zone:
        props["zone"] = zone
    async with drv.session() as session:
        await session.run(
            "MERGE (b:Bed {id: $id}) SET b += $props",
            id=bed_id,
            props=props,
        )


# ---------------------------------------------------------------------------
# Edge upserts
# ---------------------------------------------------------------------------

async def upsert_bed_assignment(
    patient_id: str,
    bed_id: str,
    since: datetime,
    *,
    driver: AsyncDriver | None = None,
) -> None:
    """
    Create an OCCUPIES edge from Patient → Bed.
    Closes any previous open OCCUPIES edge (sets `until` timestamp).
    Requirement 7.3
    """
    drv = driver or get_neo4j_driver()
    since_str = _dt(since)

    async with drv.session() as session:
        # Close previous open assignment for this patient
        await session.run(
            """
            MATCH (p:Patient {id: $patient_id})-[r:OCCUPIES]->(b:Bed)
            WHERE r.until IS NULL AND b.id <> $bed_id
            SET r.until = $since
            """,
            patient_id=patient_id,
            bed_id=bed_id,
            since=since_str,
        )
        # Ensure Bed node exists
        await session.run(
            "MERGE (b:Bed {id: $bed_id})",
            bed_id=bed_id,
        )
        # Create new OCCUPIES edge (idempotent on since)
        await session.run(
            """
            MATCH (p:Patient {id: $patient_id})
            MATCH (b:Bed {id: $bed_id})
            MERGE (p)-[r:OCCUPIES {since: $since}]->(b)
            ON CREATE SET r.until = null
            """,
            patient_id=patient_id,
            bed_id=bed_id,
            since=since_str,
        )
        # Update patient's bed_id property
        await session.run(
            "MATCH (p:Patient {id: $patient_id}) SET p.bed_id = $bed_id",
            patient_id=patient_id,
            bed_id=bed_id,
        )
    logger.debug("Upserted OCCUPIES edge %s → %s", patient_id, bed_id)


async def upsert_staff_contact(
    patient_id: str,
    staff_id: str,
    role: str,
    shift_start: datetime,
    shift_end: datetime | None = None,
    *,
    driver: AsyncDriver | None = None,
) -> None:
    """
    Create a TREATED_BY edge from Patient → Staff.
    Requirement 7.2
    """
    drv = driver or get_neo4j_driver()
    async with drv.session() as session:
        await session.run("MERGE (s:Staff {id: $staff_id})", staff_id=staff_id)
        await session.run(
            """
            MATCH (p:Patient {id: $patient_id})
            MATCH (s:Staff {id: $staff_id})
            MERGE (p)-[r:TREATED_BY {shift_start: $shift_start}]->(s)
            SET r.role = $role,
                r.shift_end = $shift_end
            """,
            patient_id=patient_id,
            staff_id=staff_id,
            role=role,
            shift_start=_dt(shift_start),
            shift_end=_dt(shift_end),
        )
    logger.debug("Upserted TREATED_BY edge %s → %s", patient_id, staff_id)


async def upsert_colocation(
    patient_a: str,
    patient_b: str,
    room_id: str,
    overlap_start: datetime,
    overlap_end: datetime | None = None,
    *,
    driver: AsyncDriver | None = None,
) -> None:
    """
    Create a CO_LOCATED edge between two patients sharing a room.
    Edge is undirected in practice — we create both directions.
    Requirement 7.2
    """
    drv = driver or get_neo4j_driver()
    async with drv.session() as session:
        await session.run(
            """
            MATCH (a:Patient {id: $patient_a})
            MATCH (b:Patient {id: $patient_b})
            MERGE (a)-[r:CO_LOCATED {room_id: $room_id, overlap_start: $overlap_start}]->(b)
            SET r.overlap_end = $overlap_end
            """,
            patient_a=patient_a,
            patient_b=patient_b,
            room_id=room_id,
            overlap_start=_dt(overlap_start),
            overlap_end=_dt(overlap_end),
        )
    logger.debug("Upserted CO_LOCATED edge %s ↔ %s in room %s", patient_a, patient_b, room_id)


async def record_pathogen_exposure(
    patient_id: str,
    pathogen: str,
    confidence: float,
    source_event_id: str,
    *,
    driver: AsyncDriver | None = None,
) -> None:
    """
    Create an EXPOSED_TO edge from Patient → Pathogen and propagate
    the exposure signal to 2-hop neighbors within 60 seconds via a
    background task.
    Requirement 7.7
    """
    drv = driver or get_neo4j_driver()
    timestamp_str = _dt(datetime.utcnow())

    async with drv.session() as session:
        # Ensure Pathogen node exists
        await session.run(
            "MERGE (pg:Pathogen {id: $pathogen})",
            pathogen=pathogen,
        )
        # Create EXPOSED_TO edge
        await session.run(
            """
            MATCH (p:Patient {id: $patient_id})
            MATCH (pg:Pathogen {id: $pathogen})
            MERGE (p)-[r:EXPOSED_TO {source_event_id: $source_event_id}]->(pg)
            SET r.confidence = $confidence,
                r.timestamp = $timestamp
            """,
            patient_id=patient_id,
            pathogen=pathogen,
            confidence=confidence,
            source_event_id=source_event_id,
            timestamp=timestamp_str,
        )

    logger.info(
        "Recorded EXPOSED_TO edge %s → %s (confidence=%.2f)",
        patient_id, pathogen, confidence,
    )

    # Propagate exposure signal to 2-hop neighbors asynchronously
    asyncio.create_task(
        _propagate_exposure(patient_id, pathogen, confidence, source_event_id, drv)
    )


async def _propagate_exposure(
    source_patient_id: str,
    pathogen: str,
    confidence: float,
    source_event_id: str,
    driver: AsyncDriver,
) -> None:
    """
    Background task: find all Patient nodes within 2 hops of the source
    patient and publish an exposure signal to the Redis stream.
    Completes within 60 seconds per Requirement 7.7.
    """
    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (src:Patient {id: $patient_id})-[*1..2]-(neighbor:Patient)
                WHERE neighbor.id <> $patient_id
                RETURN DISTINCT neighbor.id AS neighbor_id
                """,
                patient_id=source_patient_id,
            )
            neighbor_ids: list[str] = [record["neighbor_id"] async for record in result]

        if neighbor_ids:
            await publish_event(
                "onco:graph:edge_update",
                {
                    "event_type": "pathogen_exposure_propagated",
                    "source_patient_id": source_patient_id,
                    "pathogen": pathogen,
                    "confidence": confidence,
                    "source_event_id": source_event_id,
                    "affected_patient_ids": neighbor_ids,
                    "timestamp": _dt(datetime.utcnow()),
                },
            )
            logger.info(
                "Propagated exposure signal to %d neighbors of %s",
                len(neighbor_ids), source_patient_id,
            )
    except Exception as exc:
        logger.error("Exposure propagation failed for %s: %s", source_patient_id, exc)
