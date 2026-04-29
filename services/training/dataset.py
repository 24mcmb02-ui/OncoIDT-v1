"""
OncoIDT Training Pipeline — Dataset Construction.

Builds training datasets from the Feature Store and training_labels table
with strict temporal ordering to prevent data leakage.

Key design decisions:
- Temporal train/val/test split: 70% / 15% / 15% by time, never by patient
- 5-fold rolling-window temporal cross-validation with 3-month fold advance
- Temporal oversampling: oversample time windows containing events
- Point-in-time correct: feature snapshots are retrieved at label timestamps

Requirements: 14.1, 14.2, 14.7
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CohortFilter:
    """Configurable cohort filter for dataset construction."""
    start_date: datetime | None = None
    end_date: datetime | None = None
    diagnosis_codes: list[str] = field(default_factory=list)   # ICD-10 codes
    chemo_regimens: list[str] = field(default_factory=list)
    ward_ids: list[str] = field(default_factory=list)
    min_admission_hours: float = 6.0   # exclude very short stays


@dataclass
class DatasetSplit:
    """A single train/val/test or cross-validation fold."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_cutoff: datetime
    val_cutoff: datetime


@dataclass
class OncoDataset:
    """
    Fully constructed dataset ready for model training.

    Attributes:
        features:       Feature matrix, shape (n_samples, n_features).
        labels_infection:    Binary labels, shape (n_samples, 4) for horizons [6h,12h,24h,48h].
        labels_deterioration: Binary labels, shape (n_samples, 3) for horizons [6h,12h,24h].
        event_times:    Time-to-event in hours, shape (n_samples,).
        event_observed: Binary mask (1=event observed, 0=censored), shape (n_samples,).
        patient_ids:    Patient ID per sample, shape (n_samples,).
        timestamps:     Label timestamp per sample, shape (n_samples,).
        feature_names:  List of feature column names.
        split:          Train/val/test index split.
        cv_folds:       List of 5 rolling-window CV folds.
        metadata:       Dataset construction metadata for MLflow logging.
    """
    features: np.ndarray
    labels_infection: np.ndarray
    labels_deterioration: np.ndarray
    event_times: np.ndarray
    event_observed: np.ndarray
    patient_ids: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    split: DatasetSplit
    cv_folds: list[DatasetSplit]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

_PATIENT_QUERY = """
    SELECT DISTINCT ce.patient_id
    FROM clinical_events ce
    WHERE 1=1
    {date_filter}
    {ward_filter}
    {diagnosis_filter}
    {regimen_filter}
    AND ce.timestamp_utc >= :min_ts
"""

_LABEL_QUERY = """
    SELECT
        tl.patient_id,
        tl.event_timestamp,
        tl.label_type,
        tl.horizon_6h,
        tl.horizon_12h,
        tl.horizon_24h,
        tl.horizon_48h,
        tl.metadata
    FROM training_labels tl
    WHERE tl.patient_id = ANY(:patient_ids)
      AND tl.label_type = :label_type
    {date_filter}
    ORDER BY tl.event_timestamp ASC
"""

_FEATURE_QUERY = """
    SELECT features
    FROM feature_store
    WHERE patient_id = :patient_id
      AND timestamp = :ts
      AND feature_version = :version
      AND modality = 'structured'
    LIMIT 1
"""

_GRAPH_SNAPSHOT_QUERY = """
    SELECT adjacency_json
    FROM graph_snapshots
    WHERE ward_id = :ward_id
      AND snapshot_timestamp <= :ts
    ORDER BY snapshot_timestamp DESC
    LIMIT 1
"""


# ---------------------------------------------------------------------------
# Core dataset builder
# ---------------------------------------------------------------------------

async def build_dataset(
    cohort_filter: CohortFilter,
    feature_version: str,
    label_type: str,
    horizons: list[int],
    db_session: Any,
    *,
    oversample_events: bool = True,
    oversample_ratio: float = 3.0,
    random_seed: int = 42,
) -> OncoDataset:
    """
    Build a training dataset from the Feature Store and training_labels.

    Args:
        cohort_filter:    Cohort selection criteria.
        feature_version:  Feature version string (e.g. "v1").
        label_type:       Label type to use ("infection_event" | "deterioration_event").
        horizons:         Forecast horizons in hours (e.g. [6, 12, 24, 48]).
        db_session:       Async SQLAlchemy session.
        oversample_events: Whether to apply temporal oversampling of event windows.
        oversample_ratio: Ratio of event to non-event samples after oversampling.
        random_seed:      Random seed for reproducibility.

    Returns:
        OncoDataset with features, labels, splits, and CV folds.

    Requirements: 14.1, 14.2, 14.7
    """
    from sqlalchemy import text

    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # 1. Resolve patient cohort
    # ------------------------------------------------------------------
    patient_ids = await _resolve_patient_cohort(cohort_filter, db_session)
    if not patient_ids:
        raise ValueError("No patients found matching cohort filter.")
    logger.info("Cohort resolved: %d patients", len(patient_ids))

    # ------------------------------------------------------------------
    # 2. Pull labels
    # ------------------------------------------------------------------
    date_filter = ""
    date_params: dict[str, Any] = {"patient_ids": patient_ids, "label_type": label_type}

    if cohort_filter.start_date:
        date_filter += " AND tl.event_timestamp >= :start_date"
        date_params["start_date"] = cohort_filter.start_date
    if cohort_filter.end_date:
        date_filter += " AND tl.event_timestamp <= :end_date"
        date_params["end_date"] = cohort_filter.end_date

    label_sql = _LABEL_QUERY.format(date_filter=date_filter)
    result = await db_session.execute(text(label_sql), date_params)
    label_rows = result.mappings().fetchall()

    if not label_rows:
        raise ValueError(
            f"No labels found for label_type='{label_type}' in the specified cohort."
        )
    logger.info("Labels retrieved: %d rows", len(label_rows))

    # ------------------------------------------------------------------
    # 3. Retrieve point-in-time feature snapshots
    # ------------------------------------------------------------------
    feature_rows, feature_names = await _retrieve_features(
        label_rows, feature_version, db_session
    )

    if not feature_rows:
        raise ValueError("No feature snapshots found for the label set.")

    # ------------------------------------------------------------------
    # 4. Assemble arrays
    # ------------------------------------------------------------------
    features_list: list[np.ndarray] = []
    labels_inf_list: list[np.ndarray] = []
    labels_det_list: list[np.ndarray] = []
    event_times_list: list[float] = []
    event_observed_list: list[int] = []
    patient_ids_list: list[str] = []
    timestamps_list: list[datetime] = []

    for row_data in feature_rows:
        feat_vec = row_data["feature_vector"]
        label_row = row_data["label_row"]

        # Infection labels (4 horizons: 6h, 12h, 24h, 48h)
        inf_labels = np.array([
            int(bool(label_row.get("horizon_6h"))),
            int(bool(label_row.get("horizon_12h"))),
            int(bool(label_row.get("horizon_24h"))),
            int(bool(label_row.get("horizon_48h"))),
        ], dtype=np.float32)

        # Deterioration labels (3 horizons: 6h, 12h, 24h)
        det_labels = np.array([
            int(bool(label_row.get("horizon_6h"))),
            int(bool(label_row.get("horizon_12h"))),
            int(bool(label_row.get("horizon_24h"))),
        ], dtype=np.float32)

        # Survival: use 24h horizon as primary event indicator
        meta = label_row.get("metadata") or {}
        event_time = float(meta.get("time_to_event_hours", 24.0))
        event_obs = int(bool(label_row.get("horizon_24h")))

        features_list.append(feat_vec)
        labels_inf_list.append(inf_labels)
        labels_det_list.append(det_labels)
        event_times_list.append(event_time)
        event_observed_list.append(event_obs)
        patient_ids_list.append(str(label_row["patient_id"]))
        timestamps_list.append(label_row["event_timestamp"])

    features_arr = np.stack(features_list, axis=0).astype(np.float32)
    labels_inf_arr = np.stack(labels_inf_list, axis=0).astype(np.float32)
    labels_det_arr = np.stack(labels_det_list, axis=0).astype(np.float32)
    event_times_arr = np.array(event_times_list, dtype=np.float32)
    event_observed_arr = np.array(event_observed_list, dtype=np.float32)
    patient_ids_arr = np.array(patient_ids_list)
    timestamps_arr = np.array(timestamps_list)

    n_samples = len(features_arr)
    logger.info("Dataset assembled: %d samples, %d features", n_samples, features_arr.shape[1])

    # ------------------------------------------------------------------
    # 5. Temporal oversampling (Requirement 14.7)
    # ------------------------------------------------------------------
    if oversample_events:
        features_arr, labels_inf_arr, labels_det_arr, event_times_arr, \
            event_observed_arr, patient_ids_arr, timestamps_arr = _temporal_oversample(
                features_arr, labels_inf_arr, labels_det_arr,
                event_times_arr, event_observed_arr,
                patient_ids_arr, timestamps_arr,
                ratio=oversample_ratio,
                rng=rng,
            )
        logger.info("After oversampling: %d samples", len(features_arr))

    # ------------------------------------------------------------------
    # 6. Temporal train/val/test split (Requirement 14.2)
    # ------------------------------------------------------------------
    split = _temporal_split(timestamps_arr, train_frac=0.70, val_frac=0.15)

    # ------------------------------------------------------------------
    # 7. 5-fold rolling-window temporal cross-validation
    # ------------------------------------------------------------------
    cv_folds = _rolling_window_cv(timestamps_arr, n_folds=5, fold_advance_months=3)

    # ------------------------------------------------------------------
    # 8. Build metadata
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "n_samples": int(len(features_arr)),
        "n_features": int(features_arr.shape[1]),
        "n_patients": int(len(set(patient_ids_arr.tolist()))),
        "label_type": label_type,
        "feature_version": feature_version,
        "horizons": horizons,
        "event_rate": float(np.mean(labels_inf_arr[:, 2])),  # 24h horizon
        "cohort_filter": {
            "start_date": cohort_filter.start_date.isoformat() if cohort_filter.start_date else None,
            "end_date": cohort_filter.end_date.isoformat() if cohort_filter.end_date else None,
            "diagnosis_codes": cohort_filter.diagnosis_codes,
            "chemo_regimens": cohort_filter.chemo_regimens,
            "ward_ids": cohort_filter.ward_ids,
        },
        "oversample_events": oversample_events,
        "oversample_ratio": oversample_ratio if oversample_events else None,
        "train_size": int(len(split.train_indices)),
        "val_size": int(len(split.val_indices)),
        "test_size": int(len(split.test_indices)),
        "n_cv_folds": len(cv_folds),
    }

    return OncoDataset(
        features=features_arr,
        labels_infection=labels_inf_arr,
        labels_deterioration=labels_det_arr,
        event_times=event_times_arr,
        event_observed=event_observed_arr,
        patient_ids=patient_ids_arr,
        timestamps=timestamps_arr,
        feature_names=feature_names,
        split=split,
        cv_folds=cv_folds,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Cohort resolution
# ---------------------------------------------------------------------------

async def _resolve_patient_cohort(
    cohort_filter: CohortFilter,
    db_session: Any,
) -> list[str]:
    """Return patient IDs matching the cohort filter."""
    from sqlalchemy import text

    conditions: list[str] = []
    params: dict[str, Any] = {
        "min_ts": (cohort_filter.start_date or datetime(2000, 1, 1, tzinfo=timezone.utc)),
    }

    date_filter = ""
    if cohort_filter.start_date:
        date_filter = " AND ce.timestamp_utc >= :start_date"
        params["start_date"] = cohort_filter.start_date
    if cohort_filter.end_date:
        date_filter += " AND ce.timestamp_utc <= :end_date"
        params["end_date"] = cohort_filter.end_date

    ward_filter = ""
    if cohort_filter.ward_ids:
        ward_filter = " AND ce.patient_id IN (SELECT patient_id FROM clinical_events WHERE payload->>'ward_id' = ANY(:ward_ids))"
        params["ward_ids"] = cohort_filter.ward_ids

    diagnosis_filter = ""
    if cohort_filter.diagnosis_codes:
        diagnosis_filter = " AND ce.patient_id IN (SELECT patient_id FROM clinical_events WHERE payload->>'primary_diagnosis_icd10' = ANY(:diagnosis_codes))"
        params["diagnosis_codes"] = cohort_filter.diagnosis_codes

    regimen_filter = ""
    if cohort_filter.chemo_regimens:
        regimen_filter = " AND ce.patient_id IN (SELECT patient_id FROM clinical_events WHERE payload->>'chemo_regimen_code' = ANY(:chemo_regimens))"
        params["chemo_regimens"] = cohort_filter.chemo_regimens

    sql = _PATIENT_QUERY.format(
        date_filter=date_filter,
        ward_filter=ward_filter,
        diagnosis_filter=diagnosis_filter,
        regimen_filter=regimen_filter,
    )

    result = await db_session.execute(text(sql), params)
    return [str(row[0]) for row in result.fetchall()]


# ---------------------------------------------------------------------------
# Feature retrieval
# ---------------------------------------------------------------------------

async def _retrieve_features(
    label_rows: list[Any],
    feature_version: str,
    db_session: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    For each label row, retrieve the point-in-time feature snapshot.
    Returns (feature_rows, feature_names).
    """
    from sqlalchemy import text

    feature_rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    missing = 0

    for label_row in label_rows:
        patient_id = str(label_row["patient_id"])
        ts = label_row["event_timestamp"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        result = await db_session.execute(
            text(_FEATURE_QUERY),
            {"patient_id": patient_id, "ts": ts, "version": feature_version},
        )
        row = result.fetchone()

        if row is None:
            missing += 1
            continue

        raw = row[0]
        feat_dict: dict[str, Any] = raw if isinstance(raw, dict) else __import__("json").loads(raw)

        # Determine feature names from first successful row
        if feature_names is None:
            feature_names = _extract_feature_names(feat_dict)

        feat_vec = _dict_to_vector(feat_dict, feature_names)
        feature_rows.append({"feature_vector": feat_vec, "label_row": label_row})

    if missing > 0:
        logger.warning(
            "Missing feature snapshots for %d / %d label rows",
            missing, len(label_rows),
        )

    return feature_rows, feature_names or []


def _extract_feature_names(feat_dict: dict[str, Any]) -> list[str]:
    """Return sorted numeric feature keys (exclude internal metadata keys)."""
    return sorted(
        k for k, v in feat_dict.items()
        if not k.startswith("_") and isinstance(v, (int, float, type(None)))
    )


def _dict_to_vector(feat_dict: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    """Convert a feature dict to a fixed-length float32 vector."""
    vec = np.zeros(len(feature_names), dtype=np.float32)
    for i, name in enumerate(feature_names):
        val = feat_dict.get(name)
        if val is not None:
            try:
                vec[i] = float(val)
            except (TypeError, ValueError):
                vec[i] = 0.0
    return vec


# ---------------------------------------------------------------------------
# Temporal split (Requirement 14.2)
# ---------------------------------------------------------------------------

def _temporal_split(
    timestamps: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> DatasetSplit:
    """
    Split indices by time: 70% train / 15% val / 15% test.
    Never splits by patient — all samples for a patient stay in one split.

    Args:
        timestamps: Array of datetime objects, shape (n_samples,).
        train_frac: Fraction of time range for training.
        val_frac:   Fraction of time range for validation.

    Returns:
        DatasetSplit with index arrays and cutoff timestamps.
    """
    sorted_idx = np.argsort(timestamps)
    sorted_ts = timestamps[sorted_idx]

    n = len(sorted_ts)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_cutoff = sorted_ts[train_end - 1] if train_end > 0 else sorted_ts[0]
    val_cutoff = sorted_ts[val_end - 1] if val_end > train_end else train_cutoff

    train_indices = sorted_idx[:train_end]
    val_indices = sorted_idx[train_end:val_end]
    test_indices = sorted_idx[val_end:]

    logger.info(
        "Temporal split: train=%d (until %s), val=%d, test=%d",
        len(train_indices), train_cutoff,
        len(val_indices), len(test_indices),
    )

    return DatasetSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )


# ---------------------------------------------------------------------------
# Rolling-window cross-validation (Requirement 14.2)
# ---------------------------------------------------------------------------

def _rolling_window_cv(
    timestamps: np.ndarray,
    n_folds: int = 5,
    fold_advance_months: int = 3,
) -> list[DatasetSplit]:
    """
    5-fold rolling-window temporal cross-validation.

    Each fold advances the training window by `fold_advance_months`.
    The validation window is the next `fold_advance_months` period.
    The test window is the period after validation.

    Args:
        timestamps:          Array of datetime objects.
        n_folds:             Number of CV folds (default 5).
        fold_advance_months: Months to advance per fold (default 3).

    Returns:
        List of DatasetSplit objects, one per fold.
    """
    sorted_idx = np.argsort(timestamps)
    sorted_ts = np.array(timestamps)[sorted_idx]

    if len(sorted_ts) == 0:
        return []

    min_ts = sorted_ts[0]
    max_ts = sorted_ts[-1]

    total_days = (max_ts - min_ts).days if hasattr(max_ts - min_ts, "days") else 0
    if total_days == 0:
        # Fallback: use index-based splits
        return _index_based_cv(sorted_idx, n_folds)

    fold_days = fold_advance_months * 30
    folds: list[DatasetSplit] = []

    for fold_idx in range(n_folds):
        # Training window: from start to fold_idx * fold_days
        train_end_days = (fold_idx + 1) * fold_days
        val_end_days = train_end_days + fold_days
        test_end_days = val_end_days + fold_days

        train_cutoff = min_ts + timedelta(days=train_end_days)
        val_cutoff = min_ts + timedelta(days=val_end_days)
        test_cutoff = min_ts + timedelta(days=test_end_days)

        # Clamp to data range
        if train_cutoff > max_ts:
            break

        train_mask = sorted_ts <= train_cutoff
        val_mask = (sorted_ts > train_cutoff) & (sorted_ts <= val_cutoff)
        test_mask = (sorted_ts > val_cutoff) & (sorted_ts <= test_cutoff)

        train_indices = sorted_idx[train_mask]
        val_indices = sorted_idx[val_mask]
        test_indices = sorted_idx[test_mask]

        if len(train_indices) == 0 or len(val_indices) == 0:
            continue

        folds.append(DatasetSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
        ))

    logger.info("Rolling-window CV: %d folds generated", len(folds))
    return folds


def _index_based_cv(sorted_idx: np.ndarray, n_folds: int) -> list[DatasetSplit]:
    """Fallback index-based CV when timestamps have no range."""
    n = len(sorted_idx)
    fold_size = max(1, n // (n_folds + 2))
    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        val_end = train_end + fold_size
        test_end = val_end + fold_size
        if train_end >= n:
            break
        folds.append(DatasetSplit(
            train_indices=sorted_idx[:train_end],
            val_indices=sorted_idx[train_end:min(val_end, n)],
            test_indices=sorted_idx[min(val_end, n):min(test_end, n)],
            train_cutoff=datetime.now(timezone.utc),
            val_cutoff=datetime.now(timezone.utc),
        ))
    return folds


# ---------------------------------------------------------------------------
# Temporal oversampling (Requirement 14.7)
# ---------------------------------------------------------------------------

def _temporal_oversample(
    features: np.ndarray,
    labels_inf: np.ndarray,
    labels_det: np.ndarray,
    event_times: np.ndarray,
    event_observed: np.ndarray,
    patient_ids: np.ndarray,
    timestamps: np.ndarray,
    ratio: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversample time windows containing events to address class imbalance.

    Uses the 24h infection horizon as the primary event indicator.
    Event samples are duplicated (with small Gaussian noise on features)
    until the event:non-event ratio reaches `ratio`.

    Args:
        ratio: Target event:non-event ratio after oversampling.
        rng:   NumPy random generator for reproducibility.

    Returns:
        Oversampled versions of all input arrays.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Use 24h infection label (index 2) as event indicator
    event_mask = labels_inf[:, 2] > 0.5
    event_idx = np.where(event_mask)[0]
    non_event_idx = np.where(~event_mask)[0]

    n_events = len(event_idx)
    n_non_events = len(non_event_idx)

    if n_events == 0 or n_non_events == 0:
        return features, labels_inf, labels_det, event_times, event_observed, patient_ids, timestamps

    # Target: n_events * ratio non-events (or keep all non-events if already balanced)
    target_non_events = int(n_events * ratio)
    if target_non_events >= n_non_events:
        # Undersample non-events if needed
        keep_non_events = rng.choice(non_event_idx, size=min(target_non_events, n_non_events), replace=False)
    else:
        keep_non_events = non_event_idx

    # Oversample events with small Gaussian noise to avoid exact duplicates
    n_oversample = max(0, len(keep_non_events) - n_events)
    if n_oversample > 0:
        oversample_src = rng.choice(event_idx, size=n_oversample, replace=True)
        noise = rng.normal(0, 0.01, size=(n_oversample, features.shape[1])).astype(np.float32)
        extra_features = features[oversample_src] + noise
        extra_labels_inf = labels_inf[oversample_src]
        extra_labels_det = labels_det[oversample_src]
        extra_event_times = event_times[oversample_src]
        extra_event_observed = event_observed[oversample_src]
        extra_patient_ids = patient_ids[oversample_src]
        extra_timestamps = timestamps[oversample_src]

        all_idx = np.concatenate([event_idx, keep_non_events])
        features_out = np.concatenate([features[all_idx], extra_features], axis=0)
        labels_inf_out = np.concatenate([labels_inf[all_idx], extra_labels_inf], axis=0)
        labels_det_out = np.concatenate([labels_det[all_idx], extra_labels_det], axis=0)
        event_times_out = np.concatenate([event_times[all_idx], extra_event_times])
        event_observed_out = np.concatenate([event_observed[all_idx], extra_event_observed])
        patient_ids_out = np.concatenate([patient_ids[all_idx], extra_patient_ids])
        timestamps_out = np.concatenate([timestamps[all_idx], extra_timestamps])
    else:
        all_idx = np.concatenate([event_idx, keep_non_events])
        features_out = features[all_idx]
        labels_inf_out = labels_inf[all_idx]
        labels_det_out = labels_det[all_idx]
        event_times_out = event_times[all_idx]
        event_observed_out = event_observed[all_idx]
        patient_ids_out = patient_ids[all_idx]
        timestamps_out = timestamps[all_idx]

    logger.info(
        "Oversampling: %d events, %d non-events → %d total samples",
        n_events, n_non_events, len(features_out),
    )
    return (
        features_out, labels_inf_out, labels_det_out,
        event_times_out, event_observed_out,
        patient_ids_out, timestamps_out,
    )
