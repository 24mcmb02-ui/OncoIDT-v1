"""
OncoIDT Graph Engine — snapshot persistence.

Serialises the current Neo4j graph state as a compressed adjacency-list
JSON blob and stores it in the `graph_snapshots` PostgreSQL table.
Supports reconstruction of historical graph states for the Training Pipeline.
Requirement 7.4
"""
from __future__ import annotations

import json
import logging
import uuid
import zlib
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver  # type: ignore[import]
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from services.graph.graph_ops import _dt, get_neo4j_driver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

async def _export_graph(
    ward_id: str,
    driver: AsyncDriver,
) -> dict[str, Any]:
    """
    Export the current graph state for a ward as an adjacency-list dict.

    Structure:
    {
        "nodes": [{"id": ..., "labels": [...], "properties": {...}}, ...],
        "edges": [{"src": ..., "dst": ..., "type": ..., "properties": {...}}, ...],
    }
    """
    async with driver.session() as session:
        # Nodes: all nodes connected to patients in this ward
        node_result = await session.run(
            """
            MATCH (p:Patient {ward_id: $ward_id})-[*0..2]-(n)
            RETURN DISTINCT
                n.id             AS node_id,
                labels(n)        AS node_labels,
                properties(n)    AS node_props
            """,
            ward_id=ward_id,
        )
        nodes = [
            {
                "id": record["node_id"],
                "labels": list(record["node_labels"]),
                "properties": dict(record["node_props"]),
            }
            async for record in node_result
        ]

        # Edges: all relationships touching those nodes
        edge_result = await session.run(
            """
            MATCH (p:Patient {ward_id: $ward_id})-[*0..2]-(n)
            WITH collect(DISTINCT n) AS ward_nodes
            UNWIND ward_nodes AS a
            MATCH (a)-[r]->(b)
            WHERE b IN ward_nodes
            RETURN
                a.id             AS src_id,
                b.id             AS dst_id,
                type(r)          AS edge_type,
                properties(r)    AS edge_props
            """,
            ward_id=ward_id,
        )
        edges = [
            {
                "src": record["src_id"],
                "dst": record["dst_id"],
                "type": record["edge_type"],
                "properties": dict(record["edge_props"]),
            }
            async for record in edge_result
        ]

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def take_snapshot(
    ward_id: str,
    db: AsyncSession,
    *,
    snapshot_timestamp: datetime | None = None,
    driver: AsyncDriver | None = None,
) -> str:
    """
    Serialise the current graph state for `ward_id` and persist it to
    the `graph_snapshots` PostgreSQL table.

    Returns the snapshot_id (UUID string).
    Requirement 7.4
    """
    drv = driver or get_neo4j_driver()
    ts = snapshot_timestamp or datetime.utcnow()

    adjacency = await _export_graph(ward_id, drv)
    node_count = len(adjacency["nodes"])
    edge_count = len(adjacency["edges"])

    # Compress the JSON to reduce storage footprint
    raw_json = json.dumps(adjacency, default=str)
    compressed = zlib.compress(raw_json.encode(), level=6)
    # Store as hex string so it fits in JSONB as a string value
    adjacency_payload: dict[str, Any] = {
        "encoding": "zlib+hex",
        "data": compressed.hex(),
        "node_count": node_count,
        "edge_count": edge_count,
    }

    snapshot_id = str(uuid.uuid4())
    await db.execute(
        text(
            """
            INSERT INTO graph_snapshots
                (snapshot_id, ward_id, snapshot_timestamp, adjacency_json,
                 node_count, edge_count, created_at)
            VALUES
                (:snapshot_id, :ward_id, :snapshot_timestamp, :adjacency_json::jsonb,
                 :node_count, :edge_count, NOW())
            """
        ),
        {
            "snapshot_id": snapshot_id,
            "ward_id": ward_id,
            "snapshot_timestamp": ts,
            "adjacency_json": json.dumps(adjacency_payload),
            "node_count": node_count,
            "edge_count": edge_count,
        },
    )
    await db.commit()

    logger.info(
        "Graph snapshot %s taken for ward %s at %s (%d nodes, %d edges)",
        snapshot_id, ward_id, _dt(ts), node_count, edge_count,
    )
    return snapshot_id


async def reconstruct_graph_at(
    ward_id: str,
    timestamp: datetime,
    db: AsyncSession,
) -> dict[str, Any]:
    """
    Retrieve and deserialise the most recent graph snapshot for `ward_id`
    that was taken at or before `timestamp`.

    Returns the adjacency-list dict:
    {
        "nodes": [...],
        "edges": [...],
    }
    Requirement 7.4
    """
    result = await db.execute(
        text(
            """
            SELECT adjacency_json
            FROM graph_snapshots
            WHERE ward_id = :ward_id
              AND snapshot_timestamp <= :timestamp
            ORDER BY snapshot_timestamp DESC
            LIMIT 1
            """
        ),
        {"ward_id": ward_id, "timestamp": timestamp},
    )
    row = result.fetchone()
    if row is None:
        logger.warning(
            "No graph snapshot found for ward %s at or before %s",
            ward_id, _dt(timestamp),
        )
        return {"nodes": [], "edges": []}

    payload: dict[str, Any] = row[0]

    if payload.get("encoding") == "zlib+hex":
        compressed = bytes.fromhex(payload["data"])
        raw_json = zlib.decompress(compressed).decode()
        adjacency: dict[str, Any] = json.loads(raw_json)
    else:
        # Legacy uncompressed format
        adjacency = payload

    logger.debug(
        "Reconstructed graph for ward %s at %s: %d nodes, %d edges",
        ward_id, _dt(timestamp),
        len(adjacency.get("nodes", [])),
        len(adjacency.get("edges", [])),
    )
    return adjacency
