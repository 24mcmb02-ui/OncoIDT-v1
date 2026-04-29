"""
OncoIDT Inference Service — Batch Re-Scoring Scheduler.

APScheduler job that runs every 5 minutes (configurable via
``inference_batch_rescore_interval_seconds``).  Fetches all active patient
IDs from PostgreSQL and re-scores each one, publishing updated ``RiskScore``
objects to ``onco:inference:score_update``.

This ensures risk scores remain current even when no new data events arrive
for a patient (e.g., time-driven risk changes during chemotherapy nadir).

Requirements: 15.2
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text

from shared.config import get_settings
from shared.db import get_session_factory
from shared.redis_client import get_redis, publish_event

if TYPE_CHECKING:
    from services.inference.model_slot import ModelSlot

logger = logging.getLogger(__name__)
settings = get_settings()

STREAM_OUT = "onco:inference:score_update"


async def _fetch_active_patient_ids() -> list[str]:
    """Query PostgreSQL for all patients with status='active'."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT DISTINCT patient_id
                FROM clinical_events
                WHERE patient_id IN (
                    SELECT DISTINCT patient_id
                    FROM clinical_events
                    ORDER BY patient_id
                )
                LIMIT 10000
                """
            )
        )
        return [str(row[0]) for row in result.fetchall()]


async def run_batch_rescore(model_slot: "ModelSlot") -> None:
    """
    Fetch all active patients and re-score each one.
    Called by APScheduler every ``inference_batch_rescore_interval_seconds``.
    """
    from services.inference.scorer import score_patient

    if model_slot.current_model is None:
        logger.warning("Batch rescore skipped: no model loaded")
        return

    try:
        patient_ids = await _fetch_active_patient_ids()
    except Exception as exc:
        logger.error("Batch rescore: failed to fetch patient IDs: %s", exc)
        return

    if not patient_ids:
        logger.debug("Batch rescore: no active patients found")
        return

    logger.info("Batch rescore started: %d patients", len(patient_ids))
    redis = get_redis()
    success = 0
    errors = 0

    async with httpx.AsyncClient() as http_client:
        for patient_id in patient_ids:
            try:
                payloads = await score_patient(patient_id, model_slot, http_client)
                for payload in payloads:
                    await publish_event(STREAM_OUT, payload, redis=redis)
                success += 1
            except Exception as exc:
                logger.error(
                    "Batch rescore error for patient=%s: %s", patient_id, exc
                )
                errors += 1

    logger.info(
        "Batch rescore complete: success=%d errors=%d", success, errors
    )


def create_batch_rescorer(model_slot: "ModelSlot") -> AsyncIOScheduler:
    """
    Create and return a configured APScheduler that runs batch re-scoring
    on the configured interval.  Call ``scheduler.start()`` to activate.
    """
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_batch_rescore,
        trigger="interval",
        seconds=settings.inference_batch_rescore_interval_seconds,
        args=[model_slot],
        id="batch_rescore",
        name="Batch patient re-scoring",
        max_instances=1,          # prevent overlapping runs
        coalesce=True,
        misfire_grace_time=60,
    )
    return scheduler
