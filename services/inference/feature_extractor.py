"""
OncoIDT Inference Service — Feature Extraction Layer.

Fetches the feature vector for a patient from the Feature Store service
(< 100ms SLA), augments it with graph neighbourhood features from the
Graph Service, and falls back gracefully on timeout or service failure.

Requirements: 15.4, 15.5
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# In-process feature cache (patient_id → (timestamp, features))
# ---------------------------------------------------------------------------
_feature_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached(patient_id: str) -> dict[str, Any] | None:
    entry = _feature_cache.get(patient_id)
    if entry is None:
        return None
    cached_at, features = entry
    if time.monotonic() - cached_at < _CACHE_TTL_SECONDS:
        return features
    del _feature_cache[patient_id]
    return None


def _set_cache(patient_id: str, features: dict[str, Any]) -> None:
    _feature_cache[patient_id] = (time.monotonic(), features)


# ---------------------------------------------------------------------------
# Feature Store client
# ---------------------------------------------------------------------------

async def _fetch_from_feature_store(
    patient_id: str,
    as_of: datetime,
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Call GET /features/{patient_id}?as_of=<iso8601> with 100ms timeout."""
    url = f"{settings.feature_store_service_url}/features/{patient_id}"
    params = {
        "as_of": as_of.isoformat(),
        "version": settings.feature_store_default_version,
    }
    resp = await http_client.get(
        url,
        params=params,
        timeout=settings.feature_store_sla_ms / 1000.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("features", {})


# ---------------------------------------------------------------------------
# Graph Service client
# ---------------------------------------------------------------------------

async def _fetch_graph_features(
    patient_id: str,
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Pull neighbourhood features from the Graph Service:
      - co_located_active_infections: count of co-located patients with active infections
      - staff_contact_count_24h: staff contacts in the last 24h
    """
    url = f"{settings.graph_service_url}/graph/neighborhood/{patient_id}"
    params = {"k": 1, "edge_types": "CO_LOCATED,TREATED_BY"}
    resp = await http_client.get(url, params=params, timeout=2.0)
    resp.raise_for_status()
    data = resp.json()

    co_located_active = sum(
        1
        for n in data.get("neighbors", [])
        if n.get("node_type") == "Patient" and n.get("active_infection", False)
    )
    staff_contacts = sum(
        1
        for n in data.get("neighbors", [])
        if n.get("node_type") == "Staff"
    )
    return {
        "co_located_active_infections": co_located_active,
        "staff_contact_count_24h": staff_contacts,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_features(
    patient_id: str,
    as_of: datetime | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """
    Extract the full feature vector for a patient.

    Strategy:
    1. Call Feature Store (< 100ms SLA). On timeout, fall back to cache.
    2. Call Graph Service for neighbourhood features. On failure, annotate
       with ``graph_unavailable=True`` and use zero-valued graph features.

    Returns a flat dict of feature name → value, plus metadata keys:
      ``_feature_version``, ``_snapshot_id``, ``_graph_unavailable``.
    """
    effective_as_of = as_of or datetime.now(timezone.utc)
    own_client = http_client is None
    client = http_client or httpx.AsyncClient()

    try:
        # --- Feature Store ---
        features: dict[str, Any] = {}
        try:
            features = await _fetch_from_feature_store(patient_id, effective_as_of, client)
            _set_cache(patient_id, features)
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning(
                "Feature store unavailable for patient=%s (%s); using cache",
                patient_id, exc,
            )
            cached = _get_cached(patient_id)
            if cached is not None:
                features = cached
                features["_from_cache"] = True
            else:
                logger.error(
                    "No cached features for patient=%s; returning empty feature vector",
                    patient_id,
                )
                features = {}

        # --- Graph Service ---
        graph_unavailable = False
        try:
            graph_features = await _fetch_graph_features(patient_id, client)
            features.update(graph_features)
        except Exception as exc:
            logger.warning(
                "Graph service unavailable for patient=%s (%s); using defaults",
                patient_id, exc,
            )
            graph_unavailable = True
            features.update({
                "co_located_active_infections": 0,
                "staff_contact_count_24h": 0,
            })

        features["_graph_unavailable"] = graph_unavailable
        features.setdefault("_feature_version", settings.feature_store_default_version)
        return features

    finally:
        if own_client:
            await client.aclose()
