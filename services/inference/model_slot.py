"""
OncoIDT Inference Service — ModelSlot hot-swap.

Holds the currently active model and an optional shadow slot for zero-downtime
model version upgrades.

Lifecycle:
  1. A background thread polls MLflow every 30 s for a new ``production``-tagged
     version of the registered model.
  2. On detection: load the new version into the shadow slot, run a smoke-test
     inference, then atomically swap current ↔ shadow.
  3. The previous model is retained for 5 minutes before being released.
  4. The service continues serving the previous model throughout the swap.

Switchover latency guarantee: < 60 seconds (Requirement 15.6, 15.7).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# LoadedModel wrapper
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Wraps a loaded MLflow PyFunc model with its version metadata."""
    version: str
    model: Any  # mlflow.pyfunc.PyFuncModel
    loaded_at: float = field(default_factory=time.monotonic)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference. Returns a dict with risk score arrays."""
        import pandas as pd
        df = pd.DataFrame([features])
        result = self.model.predict(df)
        if isinstance(result, dict):
            return result
        # Fallback: wrap raw array output
        return {"raw": result}


def _load_model(version: str) -> LoadedModel:
    """Load a specific model version from MLflow registry."""
    import mlflow.pyfunc
    model_uri = (
        f"models:/{settings.mlflow_model_registry_name}/{version}"
    )
    logger.info("Loading model version=%s from %s", version, model_uri)
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    return LoadedModel(version=version, model=pyfunc_model)


def _smoke_test(loaded: LoadedModel) -> bool:
    """Run a minimal inference to verify the model is functional."""
    try:
        dummy_features: dict[str, Any] = {k: 0.0 for k in [
            "anc", "temperature_c", "heart_rate_bpm", "respiratory_rate_rpm",
            "sbp_mmhg", "dbp_mmhg", "spo2_pct",
            "days_since_last_chemo_dose", "immunosuppression_score",
            "co_located_active_infections", "staff_contact_count_24h",
        ]}
        result = loaded.predict(dummy_features)
        return result is not None
    except Exception as exc:
        logger.warning("Smoke test failed for version=%s: %s", loaded.version, exc)
        return False


def _get_latest_production_version() -> str | None:
    """Query MLflow for the latest ``Production``-staged version."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri
        )
        versions = client.get_latest_versions(
            settings.mlflow_model_registry_name,
            stages=["Production"],
        )
        if versions:
            return versions[0].version
    except Exception as exc:
        logger.debug("MLflow poll error: %s", exc)
    return None


# ---------------------------------------------------------------------------
# ModelSlot
# ---------------------------------------------------------------------------

class ModelSlot:
    """Thread-safe model slot with hot-swap support.

    Usage::

        slot = ModelSlot()
        slot.start_polling()          # starts background thread
        model = slot.current_model    # always returns the active model (or None)
        slot.stop_polling()
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: LoadedModel | None = None
        self._shadow: LoadedModel | None = None
        self._shadow_loaded_at: float | None = None
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> LoadedModel | None:
        with self._lock:
            return self._current

    @property
    def current_version(self) -> str | None:
        with self._lock:
            return self._current.version if self._current else None

    def load_initial(self) -> None:
        """Attempt to load the current production model at startup."""
        version = _get_latest_production_version()
        if version is None:
            logger.warning("No production model found in MLflow at startup.")
            return
        try:
            loaded = _load_model(version)
            with self._lock:
                self._current = loaded
            logger.info("Initial model loaded: version=%s", version)
        except Exception as exc:
            logger.error("Failed to load initial model version=%s: %s", version, exc)

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def start_polling(self) -> None:
        """Start the background thread that polls MLflow for new versions."""
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="model-slot-poller",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info("Model slot polling started (interval=%ds)",
                    settings.inference_model_poll_interval_seconds)

    def stop_polling(self) -> None:
        """Signal the polling thread to stop and wait for it."""
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5.0)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_for_new_version()
                self._maybe_release_shadow()
            except Exception as exc:
                logger.error("Model slot poll error: %s", exc)
            self._stop_event.wait(timeout=settings.inference_model_poll_interval_seconds)

    def _check_for_new_version(self) -> None:
        latest = _get_latest_production_version()
        if latest is None:
            return

        with self._lock:
            current_version = self._current.version if self._current else None
            shadow_version = self._shadow.version if self._shadow else None

        if latest == current_version or latest == shadow_version:
            return  # already loaded or loading

        logger.info("New production model detected: version=%s (current=%s)",
                    latest, current_version)
        self._load_and_swap(latest)

    def _load_and_swap(self, version: str) -> None:
        """Load version into shadow, smoke-test, then atomically swap."""
        try:
            candidate = _load_model(version)
        except Exception as exc:
            logger.error("Failed to load candidate version=%s: %s", version, exc)
            return

        if not _smoke_test(candidate):
            logger.error("Smoke test failed for version=%s; aborting swap", version)
            return

        with self._lock:
            old = self._current
            self._shadow = old          # retain old for 5 min
            self._shadow_loaded_at = time.monotonic()
            self._current = candidate

        logger.info(
            "Model swapped: %s → %s",
            old.version if old else "none",
            version,
        )

    def _maybe_release_shadow(self) -> None:
        """Release the shadow model after the retention window."""
        with self._lock:
            if self._shadow is None or self._shadow_loaded_at is None:
                return
            elapsed = time.monotonic() - self._shadow_loaded_at
            if elapsed >= settings.inference_model_retain_seconds:
                logger.info(
                    "Releasing shadow model version=%s after %.0fs",
                    self._shadow.version, elapsed,
                )
                self._shadow = None
                self._shadow_loaded_at = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_slot: ModelSlot | None = None


def get_model_slot() -> ModelSlot:
    """Return the process-level ModelSlot singleton."""
    global _slot
    if _slot is None:
        _slot = ModelSlot()
    return _slot
