"""
OncoIDT Inference Service — Event-Driven Scoring Loop.

Consumes ``onco:ingestion:patient_event`` from Redis Streams (consumer group
``inference-service``), runs the full inference pipeline for each event, and
publishes ``RiskScore`` objects to ``onco:inference:score_update``.

Pipeline per event:
  1. Extract features (Feature Store + Graph Service)
  2. Run Neural CDE + Graph Transformer → infection & deterioration scores
  3. Run DeepHit survival model → survival estimate
  4. Apply conformal prediction intervals
  5. Publish RiskScore to ``onco:inference:score_update``

On inference error: log with stack trace, retain last valid score with
``staleness_flag=True``, continue processing.

Requirements: 15.1, 15.3, 15.5
"""
from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np

from shared.config import get_settings
from shared.redis_client import consume_events, get_redis, publish_event
from services.inference.feature_extractor import extract_features
from services.inference.model_slot import ModelSlot

logger = logging.getLogger(__name__)
settings = get_settings()

STREAM_IN = "onco:ingestion:patient_event"
STREAM_OUT = "onco:inference:score_update"
CONSUMER_GROUP = "inference-service"
CONSUMER_NAME = "scorer-0"

# Forecast horizons (hours) for infection and deterioration
INFECTION_HORIZONS = [6, 12, 24, 48]
DETERIORATION_HORIZONS = [6, 12, 24]

# In-process stale score cache: patient_id → last published score payload
_stale_cache: dict[str, dict[str, Any]] = {}


def _build_score_payload(
    patient_id: str,
    score_type: str,
    horizon_hours: int,
    score: float,
    lower: float,
    upper: float,
    model_version: str,
    feature_version: str,
    snapshot_id: str,
    staleness_flag: bool = False,
) -> dict[str, Any]:
    return {
        "score_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "score_type": score_type,
        "forecast_horizon_hours": horizon_hours,
        "score": round(float(score), 6),
        "uncertainty_lower": round(float(lower), 6),
        "uncertainty_upper": round(float(upper), 6),
        "model_version": model_version,
        "feature_version": feature_version,
        "feature_snapshot_id": snapshot_id,
        "rule_overrides": [],
        "staleness_flag": staleness_flag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _features_to_numpy(features: dict[str, Any]) -> np.ndarray:
    """Convert a flat feature dict to a 1-D numpy array for model input."""
    ORDERED_KEYS = [
        "anc", "wbc", "lymphocytes", "crp_mg_l", "procalcitonin_ug_l",
        "temperature_c", "heart_rate_bpm", "respiratory_rate_rpm",
        "sbp_mmhg", "dbp_mmhg", "spo2_pct", "gcs",
        "days_since_last_chemo_dose", "immunosuppression_score",
        "chemo_cycle_number",
        "vitals_mean_1h", "vitals_std_1h", "vitals_min_1h", "vitals_max_1h",
        "vitals_mean_6h", "vitals_std_6h", "vitals_min_6h", "vitals_max_6h",
        "vitals_mean_24h", "vitals_std_24h", "vitals_min_24h", "vitals_max_24h",
        "anc_slope_6h", "anc_slope_24h",
        "time_since_last_antibiotic_hours",
        "prior_infection_count", "antibiotic_active",
        "co_located_active_infections", "staff_contact_count_24h",
    ]
    return np.array(
        [float(features.get(k, 0.0) or 0.0) for k in ORDERED_KEYS],
        dtype=np.float32,
    )


async def _run_inference_for_patient(
    patient_id: str,
    model_slot: ModelSlot,
    http_client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """
    Full inference pipeline for one patient.
    Returns a list of score payload dicts (one per horizon per score type).
    """
    as_of = datetime.now(timezone.utc)

    # 1. Feature extraction
    features = await extract_features(patient_id, as_of, http_client)
    feature_version = features.get("_feature_version", settings.feature_store_default_version)
    snapshot_id = features.get("_snapshot_id", str(uuid.uuid4()))

    loaded = model_slot.current_model
    if loaded is None:
        raise RuntimeError("No model loaded in ModelSlot")

    model_version = loaded.version

    # 2. Model inference via MLflow pyfunc
    result = loaded.predict(features)

    # Expected keys from the pyfunc model:
    #   infection_risk: list[float] of length 4 (horizons 6/12/24/48h)
    #   deterioration_risk: list[float] of length 3 (horizons 6/12/24h)
    #   survival_median_hours: float
    #   survival_ci_lower: float
    #   survival_ci_upper: float
    #   conformal_quantile: float  (optional — pre-calibrated)
    infection_scores = result.get("infection_risk", [0.5] * 4)
    deterioration_scores = result.get("deterioration_risk", [0.5] * 3)
    conformal_q = float(result.get("conformal_quantile", 0.1))

    payloads: list[dict[str, Any]] = []

    # 3. Infection risk scores
    for i, horizon in enumerate(INFECTION_HORIZONS):
        score = float(infection_scores[i]) if i < len(infection_scores) else 0.5
        lower = max(0.0, score - conformal_q)
        upper = min(1.0, score + conformal_q)
        payloads.append(_build_score_payload(
            patient_id, "infection", horizon, score, lower, upper,
            model_version, feature_version, snapshot_id,
        ))

    # 4. Deterioration risk scores
    for i, horizon in enumerate(DETERIORATION_HORIZONS):
        score = float(deterioration_scores[i]) if i < len(deterioration_scores) else 0.5
        lower = max(0.0, score - conformal_q)
        upper = min(1.0, score + conformal_q)
        payloads.append(_build_score_payload(
            patient_id, "deterioration", horizon, score, lower, upper,
            model_version, feature_version, snapshot_id,
        ))

    # 5. Survival estimate (published as a separate event type)
    survival_payload = {
        "patient_id": patient_id,
        "event_type": "survival_estimate",
        "median_hours": float(result.get("survival_median_hours", 48.0)),
        "ci_80_lower_hours": float(result.get("survival_ci_lower", 24.0)),
        "ci_80_upper_hours": float(result.get("survival_ci_upper", 72.0)),
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payloads.append(survival_payload)

    return payloads


async def score_patient(
    patient_id: str,
    model_slot: ModelSlot,
    http_client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """
    Score a single patient, falling back to stale scores on error.
    Always returns a list of score payloads (possibly stale).
    """
    try:
        payloads = await _run_inference_for_patient(patient_id, model_slot, http_client)
        # Cache for staleness fallback
        _stale_cache[patient_id] = payloads[-1]  # store last payload as sentinel
        return payloads
    except Exception:
        logger.error(
            "Inference error for patient=%s:\n%s",
            patient_id, traceback.format_exc(),
        )
        # Return stale scores if available
        stale = _stale_cache.get(patient_id)
        if stale:
            stale_payloads = []
            for horizon in INFECTION_HORIZONS:
                stale_payloads.append(_build_score_payload(
                    patient_id, "infection", horizon,
                    stale.get("score", 0.5),
                    stale.get("uncertainty_lower", 0.4),
                    stale.get("uncertainty_upper", 0.6),
                    stale.get("model_version", "unknown"),
                    stale.get("feature_version", settings.feature_store_default_version),
                    stale.get("feature_snapshot_id", ""),
                    staleness_flag=True,
                ))
            return stale_payloads
        return []


async def run_event_consumer(model_slot: ModelSlot) -> None:
    """
    Async task: consume ``onco:ingestion:patient_event`` and score each patient.
    Runs indefinitely; designed to be started as a FastAPI lifespan background task.
    """
    logger.info("Event consumer started on stream=%s group=%s", STREAM_IN, CONSUMER_GROUP)
    redis = get_redis()
    async with httpx.AsyncClient() as http_client:
        async for msg_id, event in consume_events(
            STREAM_IN,
            CONSUMER_GROUP,
            CONSUMER_NAME,
            redis=redis,
        ):
            patient_id = event.get("patient_id")
            if not patient_id:
                logger.warning("Received event without patient_id: %s", event)
                continue

            payloads = await score_patient(patient_id, model_slot, http_client)
            for payload in payloads:
                try:
                    await publish_event(STREAM_OUT, payload, redis=redis)
                except Exception as exc:
                    logger.error("Failed to publish score for patient=%s: %s", patient_id, exc)
