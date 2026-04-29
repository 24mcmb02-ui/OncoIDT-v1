"""
OncoIDT Neo4j schema initialisation script.

Creates uniqueness constraints and indexes required by the Graph Engine.
Safe to run multiple times — all statements use IF NOT EXISTS.

Usage:
    python infra/neo4j_schema_init.py

Environment variables (or .env file):
    NEO4J_URI      bolt://neo4j:7687
    NEO4J_USER     neo4j
    NEO4J_PASSWORD neo4j_secret
"""
from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neo4j import GraphDatabase, Driver  # type: ignore[import]
from shared.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uniqueness constraints (Requirements 7.1, 7.2)
# Each constraint also implicitly creates a backing index.
# ---------------------------------------------------------------------------
CONSTRAINTS: list[tuple[str, str]] = [
    # (constraint_name, cypher)
    (
        "constraint_patient_id",
        "CREATE CONSTRAINT constraint_patient_id IF NOT EXISTS "
        "FOR (n:Patient) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_bed_id",
        "CREATE CONSTRAINT constraint_bed_id IF NOT EXISTS "
        "FOR (n:Bed) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_staff_id",
        "CREATE CONSTRAINT constraint_staff_id IF NOT EXISTS "
        "FOR (n:Staff) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_pathogen_id",
        "CREATE CONSTRAINT constraint_pathogen_id IF NOT EXISTS "
        "FOR (n:Pathogen) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_room_id",
        "CREATE CONSTRAINT constraint_room_id IF NOT EXISTS "
        "FOR (n:Room) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_ward_id",
        "CREATE CONSTRAINT constraint_ward_id IF NOT EXISTS "
        "FOR (n:Ward) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_clinical_event_id",
        "CREATE CONSTRAINT constraint_clinical_event_id IF NOT EXISTS "
        "FOR (n:ClinicalEvent) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_lab_result_id",
        "CREATE CONSTRAINT constraint_lab_result_id IF NOT EXISTS "
        "FOR (n:LabResult) REQUIRE n.id IS UNIQUE",
    ),
]

# ---------------------------------------------------------------------------
# Additional indexes for query performance (Requirements 7.1, 7.2)
# ---------------------------------------------------------------------------
INDEXES: list[tuple[str, str]] = [
    # (index_name, cypher)
    (
        "index_patient_ward_id",
        "CREATE INDEX index_patient_ward_id IF NOT EXISTS "
        "FOR (n:Patient) ON (n.ward_id)",
    ),
    (
        "index_patient_status",
        "CREATE INDEX index_patient_status IF NOT EXISTS "
        "FOR (n:Patient) ON (n.status)",
    ),
    (
        "index_clinical_event_timestamp",
        "CREATE INDEX index_clinical_event_timestamp IF NOT EXISTS "
        "FOR (n:ClinicalEvent) ON (n.timestamp)",
    ),
    (
        "index_clinical_event_type",
        "CREATE INDEX index_clinical_event_type IF NOT EXISTS "
        "FOR (n:ClinicalEvent) ON (n.event_type)",
    ),
    (
        "index_lab_result_loinc",
        "CREATE INDEX index_lab_result_loinc IF NOT EXISTS "
        "FOR (n:LabResult) ON (n.loinc_code)",
    ),
    (
        "index_bed_room_id",
        "CREATE INDEX index_bed_room_id IF NOT EXISTS "
        "FOR (n:Bed) ON (n.room_id)",
    ),
    (
        "index_staff_role",
        "CREATE INDEX index_staff_role IF NOT EXISTS "
        "FOR (n:Staff) ON (n.role)",
    ),
]


def run_schema_init(driver: Driver) -> None:
    """Apply all constraints and indexes against the default database."""
    with driver.session() as session:
        logger.info("Applying uniqueness constraints...")
        for name, cypher in CONSTRAINTS:
            try:
                session.run(cypher)
                logger.info("  ✓ %s", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  ! %s — %s", name, exc)

        logger.info("Applying indexes...")
        for name, cypher in INDEXES:
            try:
                session.run(cypher)
                logger.info("  ✓ %s", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  ! %s — %s", name, exc)

    logger.info("Neo4j schema initialisation complete.")


def main() -> None:
    settings = get_settings()
    logger.info(
        "Connecting to Neo4j at %s as %s", settings.neo4j_uri, settings.neo4j_user
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        run_schema_init(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
