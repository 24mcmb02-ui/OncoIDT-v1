"""
OncoIDT Explainability Service — Global Ward Explanation Aggregation.

Aggregates SHAP values across all active patients every 30 minutes via an
APScheduler job and stores the result in the ``ward_explanations`` table.

Requirements: 11.6
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from services.explainability.shap_engine import FEATURE_NAMES, FeatureAttribution
from shared.config import get_settings
from shared.db import get_session_factory

logger = logging.getLogger(__name__)
settings = get_settings()

# How many top features to store in the global ward explanation
GLOBAL_TOP_N = 10


async def _fetch_active_patient_ids(session: AsyncSession, ward_id: str) -> list[str]:
    """Return patient IDs for all active patients in a ward."""
    result = await session.execute(
        text(
            """
            SELECT DISTINCT patient_id
            FROM clinical_events
            WHERE (payload->>'ward_id' = :ward_id OR :ward_id = 'all')
              AND timestamp_utc >= NOW() - INTERVAL '24 hours'
            LIMIT 500
            """
        ),
        {"ward_id": ward_id},
    )
    return [str(row[0]) for row in result.fetchall()]


async def _fetch_recent_shap_values(
    session: AsyncSession,
    patient_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Fetch the most recent SHAP explanation for each patient from the
    ``explanations`` table.
    """
    if not patient_ids:
        return []

    result = await session.execute(
        text(
            """
            SELECT DISTINCT ON (patient_id)
                patient_id, shap_values, feature_names, score_type
            FROM explanations
            WHERE patient_id = ANY(:patient_ids)
            ORDER BY patient_id, computed_at DESC
            """
        ),
        {"patient_ids": patient_ids},
    )
    rows = result.fetchall()
    return [
        {
            "patient_id": str(row[0]),
            "shap_values": json.loads(row[1]) if isinstance(row[1], str) else row[1],
            "feature_names": json.loads(row[2]) if isinstance(row[2], str) else row[2],
            "score_type": str(row[3]),
        }
        for row in rows
    ]


def _aggregate_shap_values(
    explanations: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate SHAP values across patients by computing the mean absolute
    SHAP value per feature.

    Returns:
        Dict mapping feature_name → mean absolute SHAP value.
    """
    if not explanations:
        return {}

    # Accumulate absolute SHAP values per feature
    feature_totals: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    for exp in explanations:
        shap_vals = exp.get("shap_values", [])
        feat_names = exp.get("feature_names", FEATURE_NAMES)

        for i, fname in enumerate(feat_names):
            if i < len(shap_vals) and fname in feature_totals:
                feature_totals[fname].append(abs(float(shap_vals[i])))

    return {
        fname: float(np.mean(vals)) if vals else 0.0
        for fname, vals in feature_totals.items()
    }


def _build_top_features(
    aggregated: dict[str, float],
    n: int = GLOBAL_TOP_N,
) -> list[dict[str, Any]]:
    """
    Return the top-N features sorted by mean absolute SHAP value.

    Returns:
        List of dicts with feature_name, mean_abs_shap, rank.
    """
    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "feature_name": fname,
            "mean_abs_shap": round(val, 6),
            "rank": rank,
        }
        for rank, (fname, val) in enumerate(sorted_features[:n], start=1)
    ]


async def compute_ward_explanation(ward_id: str) -> dict[str, Any] | None:
    """
    Compute and persist the global ward explanation for a given ward.

    Fetches recent SHAP values for all active patients, aggregates them,
    and stores the result in ``ward_explanations``.

    Args:
        ward_id: Ward identifier, or "all" to aggregate across all wards.

    Returns:
        The persisted ward explanation dict, or None on failure.
    """
    session_factory = get_session_factory()
    computed_at = datetime.now(timezone.utc)

    try:
        async with session_factory() as session:
            # 1. Fetch active patient IDs for this ward
            patient_ids = await _fetch_active_patient_ids(session, ward_id)
            if not patient_ids:
                logger.info(
                    "No active patients found for ward=%s; skipping global explanation.",
                    ward_id,
                )
                return None

            # 2. Fetch recent SHAP values
            explanations = await _fetch_recent_shap_values(session, patient_ids)
            if not explanations:
                logger.info(
                    "No SHAP explanations found for ward=%s patients; skipping.",
                    ward_id,
                )
                return None

            # 3. Aggregate
            aggregated = _aggregate_shap_values(explanations)
            top_features = _build_top_features(aggregated)

            # 4. Persist to ward_explanations table
            explanation_id = str(uuid.uuid4())
            payload = {
                "top_features": top_features,
                "patient_count": len(explanations),
                "feature_count": len(aggregated),
            }

            await session.execute(
                text(
                    """
                    INSERT INTO ward_explanations
                        (explanation_id, ward_id, computed_at, patient_count,
                         top_features, aggregated_shap)
                    VALUES
                        (:explanation_id, :ward_id, :computed_at, :patient_count,
                         :top_features::jsonb, :aggregated_shap::jsonb)
                    ON CONFLICT (ward_id, computed_at) DO UPDATE SET
                        patient_count = EXCLUDED.patient_count,
                        top_features = EXCLUDED.top_features,
                        aggregated_shap = EXCLUDED.aggregated_shap
                    """
                ),
                {
                    "explanation_id": explanation_id,
                    "ward_id": ward_id,
                    "computed_at": computed_at,
                    "patient_count": len(explanations),
                    "top_features": json.dumps(top_features),
                    "aggregated_shap": json.dumps(
                        {k: round(v, 6) for k, v in aggregated.items()}
                    ),
                },
            )
            await session.commit()

            logger.info(
                "Global ward explanation computed for ward=%s: %d patients, "
                "top feature=%s (mean_abs_shap=%.4f)",
                ward_id,
                len(explanations),
                top_features[0]["feature_name"] if top_features else "N/A",
                top_features[0]["mean_abs_shap"] if top_features else 0.0,
            )

            return {
                "explanation_id": explanation_id,
                "ward_id": ward_id,
                "computed_at": computed_at.isoformat(),
                **payload,
            }

    except Exception:
        logger.exception("Failed to compute global ward explanation for ward=%s", ward_id)
        return None


async def _run_global_explanation_job() -> None:
    """
    APScheduler job: compute global explanations for all known wards.

    Queries the database for distinct ward IDs with active patients and
    runs compute_ward_explanation() for each.
    """
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT DISTINCT payload->>'ward_id' AS ward_id
                    FROM clinical_events
                    WHERE timestamp_utc >= NOW() - INTERVAL '24 hours'
                      AND payload->>'ward_id' IS NOT NULL
                    """
                )
            )
            ward_ids = [str(row[0]) for row in result.fetchall() if row[0]]
    except Exception:
        logger.exception("Failed to fetch ward IDs for global explanation job")
        return

    for ward_id in ward_ids:
        await compute_ward_explanation(ward_id)


def create_global_explanation_scheduler() -> AsyncIOScheduler:
    """
    Create and return an APScheduler that runs the global ward explanation
    aggregation job every 30 minutes.

    The caller is responsible for calling scheduler.start() and
    scheduler.shutdown().

    Requirements: 11.6
    """
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _run_global_explanation_job,
        trigger="interval",
        minutes=30,
        id="global-ward-explanation",
        name="Global Ward Explanation Aggregation",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler
