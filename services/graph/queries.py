"""
OncoIDT Graph Engine — query interface.

Provides k-hop neighborhood queries, infection signal propagation queries,
and contact tracing over the Neo4j patient-ward knowledge graph.
Requirement 7.6
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver  # type: ignore[import]

from services.graph.graph_ops import _dt, get_neo4j_driver

logger = logging.getLogger(__name__)


async def get_k_hop_neighborhood(
    node_id: str,
    k: int,
    edge_types: list[str] | None = None,
    since: datetime | None = None,
    *,
    driver: AsyncDriver | None = None,
) -> list[dict[str, Any]]:
    """
    Return all nodes within k hops of `node_id`, optionally filtered by
    edge type(s) and a minimum timestamp on the relationship.

    Args:
        node_id:    ID of the source node (Patient, Bed, Staff, etc.)
        k:          Maximum hop distance (1 or 2 recommended for performance)
        edge_types: List of relationship type names to traverse, e.g.
                    ["CO_LOCATED", "TREATED_BY"].  None means all types.
        since:      Only traverse edges whose `since` / `overlap_start` /
                    `shift_start` / `timestamp` is >= this value.
        driver:     Optional injected driver (for testing).

    Returns:
        List of dicts with keys: node_id, node_labels, properties, hops.
    """
    drv = driver or get_neo4j_driver()
    since_str = _dt(since)

    # Build relationship type filter
    if edge_types:
        rel_filter = "|".join(edge_types)
        rel_pattern = f"[r:{rel_filter}*1..{k}]"
    else:
        rel_pattern = f"[*1..{k}]"

    # Time filter clause — applied as a WHERE on the path if since is given
    time_clause = ""
    if since_str:
        time_clause = (
            "AND ALL(rel IN relationships(path) WHERE "
            "  coalesce(rel.since, rel.overlap_start, rel.shift_start, rel.timestamp, $since) >= $since"
            ")"
        )

    cypher = f"""
        MATCH path = (src {{id: $node_id}})-{rel_pattern}-(neighbor)
        WHERE neighbor.id <> $node_id
        {time_clause}
        RETURN DISTINCT
            neighbor.id          AS node_id,
            labels(neighbor)     AS node_labels,
            properties(neighbor) AS properties,
            length(path)         AS hops
        ORDER BY hops ASC
    """

    async with drv.session() as session:
        result = await session.run(
            cypher,
            node_id=node_id,
            since=since_str,
        )
        rows = [
            {
                "node_id": record["node_id"],
                "node_labels": list(record["node_labels"]),
                "properties": dict(record["properties"]),
                "hops": record["hops"],
            }
            async for record in result
        ]

    logger.debug(
        "k-hop query for %s (k=%d, edge_types=%s) → %d nodes",
        node_id, k, edge_types, len(rows),
    )
    return rows


async def get_infection_signal_neighbors(
    patient_id: str,
    *,
    driver: AsyncDriver | None = None,
) -> list[str]:
    """
    Return patient IDs of all neighbors within 2 hops that have an active
    infection flag (infection_risk_score > 0.6 or confirmed_infection=True).
    Requirement 7.6
    """
    drv = driver or get_neo4j_driver()

    async with drv.session() as session:
        result = await session.run(
            """
            MATCH (src:Patient {id: $patient_id})-[*1..2]-(neighbor:Patient)
            WHERE neighbor.id <> $patient_id
              AND (
                neighbor.infection_risk_score > 0.6
                OR neighbor.confirmed_infection = true
              )
            RETURN DISTINCT neighbor.id AS neighbor_id
            """,
            patient_id=patient_id,
        )
        neighbor_ids = [record["neighbor_id"] async for record in result]

    logger.debug(
        "Infection signal neighbors for %s: %d found",
        patient_id, len(neighbor_ids),
    )
    return neighbor_ids


async def get_contact_trace(
    patient_id: str,
    since: datetime,
    *,
    driver: AsyncDriver | None = None,
) -> list[dict[str, Any]]:
    """
    Return the full CO_LOCATED + TREATED_BY contact chain for a patient
    since the given timestamp.  Used for infection contact tracing.
    Requirement 7.6
    """
    drv = driver or get_neo4j_driver()
    since_str = _dt(since)

    async with drv.session() as session:
        result = await session.run(
            """
            MATCH (src:Patient {id: $patient_id})-[r:CO_LOCATED|TREATED_BY*1..2]-(contact)
            WHERE coalesce(
                r[-1].overlap_start,
                r[-1].shift_start,
                r[-1].since,
                $since
            ) >= $since
            RETURN DISTINCT
                contact.id          AS contact_id,
                labels(contact)     AS contact_labels,
                properties(contact) AS contact_props,
                type(r[-1])         AS edge_type
            """,
            patient_id=patient_id,
            since=since_str,
        )
        contacts = [
            {
                "contact_id": record["contact_id"],
                "contact_labels": list(record["contact_labels"]),
                "contact_props": dict(record["contact_props"]),
                "edge_type": record["edge_type"],
            }
            async for record in result
        ]

    logger.debug(
        "Contact trace for %s since %s: %d contacts",
        patient_id, since_str, len(contacts),
    )
    return contacts
