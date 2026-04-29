"""
OncoIDT Feature Store — point-in-time feature computation functions.

All functions accept `as_of: datetime` and query ONLY data with
`timestamp_utc <= as_of`, guaranteeing no future leakage (Requirement 19.2).

Functions return plain dicts so they can be serialised directly to JSONB
in the feature_store table.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Conn = AsyncSession | AsyncConnection
FeatureDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def _scalar(conn: Conn, sql: str, params: dict[str, Any]) -> Any:
    """Execute a scalar query and return the single value (or None)."""
    result = await conn.execute(text(sql), params)
    row = result.fetchone()
    return row[0] if row else None


async def _row(conn: Conn, sql: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Execute a query and return the first row as a dict (or None)."""
    result = await conn.execute(text(sql), params)
    row = result.mappings().fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# 1. Vitals features
# ---------------------------------------------------------------------------

async def compute_vitals_features(
    patient_id: str,
    as_of: datetime,
    conn: Conn,
) -> FeatureDict:
    """
    Latest vitals snapshot + rolling stats (mean, std, min, max) over
    1h, 6h, 24h windows.  Uses TimescaleDB time_bucket aggregates where
    available; falls back to plain window queries.

    Point-in-time correct: only records with timestamp_utc <= as_of are used.
    """
    as_of = _ensure_utc(as_of)
    params: dict[str, Any] = {"patient_id": patient_id, "as_of": as_of}

    # Latest vitals record
    latest_sql = """
        SELECT
            payload->>'temperature_c'        AS temperature_c,
            payload->>'heart_rate_bpm'        AS heart_rate_bpm,
            payload->>'respiratory_rate_rpm'  AS respiratory_rate_rpm,
            payload->>'sbp_mmhg'              AS sbp_mmhg,
            payload->>'dbp_mmhg'              AS dbp_mmhg,
            payload->>'spo2_pct'              AS spo2_pct,
            payload->>'gcs'                   AS gcs,
            timestamp_utc
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'vital'
          AND timestamp_utc <= :as_of
        ORDER BY timestamp_utc DESC
        LIMIT 1
    """
    latest = await _row(conn, latest_sql, params)

    def _f(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    features: FeatureDict = {
        "vitals_latest_temperature_c": _f(latest.get("temperature_c")) if latest else None,
        "vitals_latest_heart_rate_bpm": _f(latest.get("heart_rate_bpm")) if latest else None,
        "vitals_latest_respiratory_rate_rpm": _f(latest.get("respiratory_rate_rpm")) if latest else None,
        "vitals_latest_sbp_mmhg": _f(latest.get("sbp_mmhg")) if latest else None,
        "vitals_latest_dbp_mmhg": _f(latest.get("dbp_mmhg")) if latest else None,
        "vitals_latest_spo2_pct": _f(latest.get("spo2_pct")) if latest else None,
        "vitals_latest_gcs": _f(latest.get("gcs")) if latest else None,
    }

    # Rolling stats per vital sign over each window
    vital_fields = [
        ("temperature_c", "temperature_c"),
        ("heart_rate_bpm", "heart_rate_bpm"),
        ("respiratory_rate_rpm", "respiratory_rate_rpm"),
        ("sbp_mmhg", "sbp_mmhg"),
        ("dbp_mmhg", "dbp_mmhg"),
        ("spo2_pct", "spo2_pct"),
    ]
    windows = [("1h", "1 hour"), ("6h", "6 hours"), ("24h", "24 hours")]

    for field_key, field_name in vital_fields:
        for win_label, win_interval in windows:
            stats_sql = f"""
                SELECT
                    AVG((payload->>'{field_name}')::float)    AS mean_val,
                    STDDEV((payload->>'{field_name}')::float) AS std_val,
                    MIN((payload->>'{field_name}')::float)    AS min_val,
                    MAX((payload->>'{field_name}')::float)    AS max_val,
                    COUNT(*)                                   AS count_val
                FROM clinical_events
                WHERE patient_id = :patient_id
                  AND record_type = 'vital'
                  AND timestamp_utc <= :as_of
                  AND timestamp_utc >= :as_of - INTERVAL '{win_interval}'
                  AND payload->>'{field_name}' IS NOT NULL
            """
            row = await _row(conn, stats_sql, params)
            prefix = f"vitals_{field_key}_{win_label}"
            if row:
                features[f"{prefix}_mean"] = _f(row.get("mean_val"))
                features[f"{prefix}_std"] = _f(row.get("std_val"))
                features[f"{prefix}_min"] = _f(row.get("min_val"))
                features[f"{prefix}_max"] = _f(row.get("max_val"))
                features[f"{prefix}_count"] = int(row.get("count_val") or 0)
            else:
                features[f"{prefix}_mean"] = None
                features[f"{prefix}_std"] = None
                features[f"{prefix}_min"] = None
                features[f"{prefix}_max"] = None
                features[f"{prefix}_count"] = 0

    return features


# ---------------------------------------------------------------------------
# 2. Lab features
# ---------------------------------------------------------------------------

async def compute_lab_features(
    patient_id: str,
    as_of: datetime,
    conn: Conn,
) -> FeatureDict:
    """
    ANC trend slope over 6h/24h, latest lab values, min/max over windows.
    Point-in-time correct: only records with timestamp_utc <= as_of are used.
    """
    as_of = _ensure_utc(as_of)
    params: dict[str, Any] = {"patient_id": patient_id, "as_of": as_of}

    def _f(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Latest lab values
    latest_sql = """
        SELECT
            payload->>'loinc_code'        AS loinc_code,
            payload->>'value_numeric'     AS value_numeric,
            timestamp_utc
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'lab'
          AND timestamp_utc <= :as_of
        ORDER BY timestamp_utc DESC
        LIMIT 50
    """
    result = await conn.execute(text(latest_sql), params)
    rows = result.mappings().fetchall()

    # LOINC codes for key lab values
    # ANC: 26499-4, WBC: 6690-2, CRP: 1988-5, Procalcitonin: 75241-0
    loinc_map = {
        "26499-4": "anc",
        "6690-2": "wbc",
        "30180-4": "lymphocytes",
        "1988-5": "crp_mg_l",
        "75241-0": "procalcitonin_ug_l",
        "2524-7": "lactate_mmol_l",
        "2160-0": "creatinine_umol_l",
    }

    latest_by_loinc: dict[str, float | None] = {}
    for row in rows:
        loinc = row.get("loinc_code", "")
        if loinc in loinc_map:
            name = loinc_map[loinc]
            if name not in latest_by_loinc:
                latest_by_loinc[name] = _f(row.get("value_numeric"))

    features: FeatureDict = {
        f"lab_latest_{name}": latest_by_loinc.get(name)
        for name in loinc_map.values()
    }

    # ANC trend slope over 6h and 24h using linear regression via SQL
    for win_label, win_interval in [("6h", "6 hours"), ("24h", "24 hours")]:
        slope_sql = f"""
            WITH anc_series AS (
                SELECT
                    EXTRACT(EPOCH FROM timestamp_utc) AS t,
                    (payload->>'value_numeric')::float AS v
                FROM clinical_events
                WHERE patient_id = :patient_id
                  AND record_type = 'lab'
                  AND payload->>'loinc_code' = '26499-4'
                  AND timestamp_utc <= :as_of
                  AND timestamp_utc >= :as_of - INTERVAL '{win_interval}'
                  AND payload->>'value_numeric' IS NOT NULL
            ),
            stats AS (
                SELECT
                    COUNT(*) AS n,
                    SUM(t) AS sum_t,
                    SUM(v) AS sum_v,
                    SUM(t * v) AS sum_tv,
                    SUM(t * t) AS sum_t2
                FROM anc_series
            )
            SELECT
                CASE
                    WHEN n >= 2 AND (n * sum_t2 - sum_t * sum_t) <> 0
                    THEN (n * sum_tv - sum_t * sum_v) / (n * sum_t2 - sum_t * sum_t)
                    ELSE NULL
                END AS slope
            FROM stats
        """
        slope = await _scalar(conn, slope_sql, params)
        # Convert slope from per-second to per-hour for interpretability
        features[f"lab_anc_slope_{win_label}_per_hour"] = (
            float(slope) * 3600.0 if slope is not None else None
        )

    # Min/max for ANC over 6h and 24h
    for win_label, win_interval in [("6h", "6 hours"), ("24h", "24 hours")]:
        minmax_sql = f"""
            SELECT
                MIN((payload->>'value_numeric')::float) AS min_val,
                MAX((payload->>'value_numeric')::float) AS max_val,
                COUNT(*) AS count_val
            FROM clinical_events
            WHERE patient_id = :patient_id
              AND record_type = 'lab'
              AND payload->>'loinc_code' = '26499-4'
              AND timestamp_utc <= :as_of
              AND timestamp_utc >= :as_of - INTERVAL '{win_interval}'
              AND payload->>'value_numeric' IS NOT NULL
        """
        row = await _row(conn, minmax_sql, params)
        features[f"lab_anc_min_{win_label}"] = _f(row.get("min_val")) if row else None
        features[f"lab_anc_max_{win_label}"] = _f(row.get("max_val")) if row else None
        features[f"lab_anc_count_{win_label}"] = int(row.get("count_val") or 0) if row else 0

    return features


# ---------------------------------------------------------------------------
# 3. Chemotherapy features
# ---------------------------------------------------------------------------

async def compute_chemo_features(
    patient_id: str,
    as_of: datetime,
    conn: Conn,
) -> FeatureDict:
    """
    Cycle number, days since last dose, cumulative dose, cycle phase,
    immunosuppression score derived from medication records.
    Point-in-time correct: only records with timestamp_utc <= as_of are used.
    """
    as_of = _ensure_utc(as_of)
    params: dict[str, Any] = {"patient_id": patient_id, "as_of": as_of}

    # Latest chemotherapy medication record
    latest_chemo_sql = """
        SELECT
            payload->>'chemo_regimen_code'          AS chemo_regimen_code,
            (payload->>'dose_mg')::float            AS dose_mg,
            payload->>'administration_timestamp'    AS admin_ts,
            timestamp_utc
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'medication'
          AND (payload->>'is_chemotherapy')::boolean = true
          AND timestamp_utc <= :as_of
        ORDER BY timestamp_utc DESC
        LIMIT 1
    """
    latest = await _row(conn, latest_chemo_sql, params)

    features: FeatureDict = {
        "chemo_regimen_code": None,
        "chemo_days_since_last_dose": None,
        "chemo_cumulative_dose_mg": None,
        "chemo_cycle_number": None,
        "chemo_cycle_phase": None,
        "chemo_immunosuppression_score": None,
    }

    if latest:
        features["chemo_regimen_code"] = latest.get("chemo_regimen_code")

        # Days since last dose
        admin_ts_str = latest.get("admin_ts")
        if admin_ts_str:
            try:
                admin_ts = datetime.fromisoformat(str(admin_ts_str))
                if admin_ts.tzinfo is None:
                    admin_ts = admin_ts.replace(tzinfo=timezone.utc)
                delta_days = (as_of - admin_ts).total_seconds() / 86400.0
                features["chemo_days_since_last_dose"] = round(delta_days, 2)
            except (ValueError, TypeError):
                pass

    # Cumulative dose (sum of all chemo doses up to as_of)
    cumulative_sql = """
        SELECT COALESCE(SUM((payload->>'dose_mg')::float), 0) AS total_dose
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'medication'
          AND (payload->>'is_chemotherapy')::boolean = true
          AND timestamp_utc <= :as_of
          AND payload->>'dose_mg' IS NOT NULL
    """
    total_dose = await _scalar(conn, cumulative_sql, params)
    features["chemo_cumulative_dose_mg"] = float(total_dose) if total_dose is not None else 0.0

    # Cycle number: count distinct chemo cycles (approximated by counting
    # regimen-start events or falling back to dose count grouping)
    cycle_count_sql = """
        SELECT COUNT(DISTINCT DATE_TRUNC('day', timestamp_utc)) AS cycle_approx
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'event'
          AND payload->>'event_type' = 'chemo_cycle_start'
          AND timestamp_utc <= :as_of
    """
    cycle_count = await _scalar(conn, cycle_count_sql, params)
    features["chemo_cycle_number"] = int(cycle_count) if cycle_count else 0

    # Cycle phase: derived from days since last dose
    days = features.get("chemo_days_since_last_dose")
    if days is not None:
        if days <= 2:
            phase = "pre"
        elif days <= 10:
            phase = "nadir"
        elif days <= 21:
            phase = "recovery"
        else:
            phase = "off"
    else:
        phase = None
    features["chemo_cycle_phase"] = phase

    # Immunosuppression score: composite [0, 1] based on days since dose
    # and cumulative dose (simplified heuristic; real model uses regimen-specific curves)
    if days is not None:
        # Peak immunosuppression around day 7-10 post-chemo
        import math
        nadir_peak = math.exp(-0.5 * ((days - 8.0) / 4.0) ** 2)
        # Scale by cumulative dose (normalised to 1000 mg reference)
        dose_factor = min(1.0, features["chemo_cumulative_dose_mg"] / 1000.0)
        features["chemo_immunosuppression_score"] = round(
            min(1.0, nadir_peak * (0.5 + 0.5 * dose_factor)), 4
        )
    else:
        features["chemo_immunosuppression_score"] = 0.0

    return features


# ---------------------------------------------------------------------------
# 4. Infection history features
# ---------------------------------------------------------------------------

async def compute_infection_history_features(
    patient_id: str,
    as_of: datetime,
    conn: Conn,
) -> FeatureDict:
    """
    Prior infection events count, days since last antibiotic, antibiotic_active flag.
    Point-in-time correct: only records with timestamp_utc <= as_of are used.
    """
    as_of = _ensure_utc(as_of)
    params: dict[str, Any] = {"patient_id": patient_id, "as_of": as_of}

    features: FeatureDict = {
        "infection_prior_events_count": 0,
        "infection_days_since_last_antibiotic": None,
        "infection_antibiotic_active": False,
        "infection_last_confirmed_timestamp": None,
    }

    # Count prior confirmed infection events
    infection_count_sql = """
        SELECT COUNT(*) AS cnt
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'event'
          AND payload->>'event_type' = 'infection_confirmed'
          AND timestamp_utc <= :as_of
    """
    cnt = await _scalar(conn, infection_count_sql, params)
    features["infection_prior_events_count"] = int(cnt) if cnt else 0

    # Last confirmed infection timestamp
    last_infection_sql = """
        SELECT timestamp_utc
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'event'
          AND payload->>'event_type' = 'infection_confirmed'
          AND timestamp_utc <= :as_of
        ORDER BY timestamp_utc DESC
        LIMIT 1
    """
    last_inf_ts = await _scalar(conn, last_infection_sql, params)
    if last_inf_ts:
        features["infection_last_confirmed_timestamp"] = (
            last_inf_ts.isoformat() if hasattr(last_inf_ts, "isoformat") else str(last_inf_ts)
        )

    # Last antibiotic administration
    # Antibiotic RxNorm codes are identified by a flag in the payload
    # (non-chemo medications with drug class 'antibiotic')
    last_abx_sql = """
        SELECT
            payload->>'administration_timestamp' AS admin_ts,
            timestamp_utc
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'medication'
          AND (payload->>'is_chemotherapy')::boolean = false
          AND payload->>'drug_class' = 'antibiotic'
          AND timestamp_utc <= :as_of
        ORDER BY timestamp_utc DESC
        LIMIT 1
    """
    last_abx = await _row(conn, last_abx_sql, params)

    if last_abx:
        admin_ts_str = last_abx.get("admin_ts") or last_abx.get("timestamp_utc")
        if admin_ts_str:
            try:
                admin_ts = datetime.fromisoformat(str(admin_ts_str))
                if admin_ts.tzinfo is None:
                    admin_ts = admin_ts.replace(tzinfo=timezone.utc)
                delta_days = (as_of - admin_ts).total_seconds() / 86400.0
                features["infection_days_since_last_antibiotic"] = round(delta_days, 2)
                # Consider antibiotic active if administered within last 2 days
                features["infection_antibiotic_active"] = delta_days <= 2.0
            except (ValueError, TypeError):
                pass

    return features


# ---------------------------------------------------------------------------
# 5. Ward context features
# ---------------------------------------------------------------------------

async def compute_ward_context_features(
    patient_id: str,
    as_of: datetime,
    conn: Conn,
) -> FeatureDict:
    """
    Room occupancy, co-located patients with active infections count,
    staff contact count in last 24h.
    Point-in-time correct: only records with timestamp_utc <= as_of are used.

    Note: full graph-derived features (co-location, staff contacts) are
    queried from clinical_events admission/transfer records here.
    The graph service provides richer versions at inference time.
    """
    as_of = _ensure_utc(as_of)
    params: dict[str, Any] = {"patient_id": patient_id, "as_of": as_of}

    features: FeatureDict = {
        "ward_room_id": None,
        "ward_bed_id": None,
        "ward_room_occupancy": 0,
        "ward_colocated_active_infections": 0,
        "ward_staff_contact_count_24h": 0,
        "ward_exposure_flag": False,
    }

    # Current bed/room assignment (latest admission or transfer event)
    bed_sql = """
        SELECT
            payload->>'bed_id'  AS bed_id,
            payload->>'room_id' AS room_id,
            payload->>'ward_id' AS ward_id
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'event'
          AND payload->>'event_type' IN ('admission', 'transfer')
          AND timestamp_utc <= :as_of
        ORDER BY timestamp_utc DESC
        LIMIT 1
    """
    bed_row = await _row(conn, bed_sql, params)

    if not bed_row:
        return features

    room_id = bed_row.get("room_id")
    bed_id = bed_row.get("bed_id")
    ward_id = bed_row.get("ward_id")

    features["ward_room_id"] = room_id
    features["ward_bed_id"] = bed_id

    if not room_id:
        return features

    room_params = {**params, "room_id": room_id, "ward_id": ward_id}

    # Room occupancy: count patients currently in the same room
    occupancy_sql = """
        WITH latest_assignments AS (
            SELECT DISTINCT ON (patient_id)
                patient_id,
                payload->>'room_id' AS room_id,
                payload->>'event_type' AS event_type
            FROM clinical_events
            WHERE record_type = 'event'
              AND payload->>'event_type' IN ('admission', 'transfer', 'discharge')
              AND timestamp_utc <= :as_of
            ORDER BY patient_id, timestamp_utc DESC
        )
        SELECT COUNT(*) AS occupancy
        FROM latest_assignments
        WHERE room_id = :room_id
          AND event_type != 'discharge'
    """
    occupancy = await _scalar(conn, occupancy_sql, room_params)
    features["ward_room_occupancy"] = int(occupancy) if occupancy else 0

    # Co-located patients with active infections (infection event in last 72h)
    colocated_infections_sql = """
        WITH room_patients AS (
            SELECT DISTINCT ON (patient_id)
                patient_id,
                payload->>'room_id' AS room_id,
                payload->>'event_type' AS event_type
            FROM clinical_events
            WHERE record_type = 'event'
              AND payload->>'event_type' IN ('admission', 'transfer', 'discharge')
              AND timestamp_utc <= :as_of
            ORDER BY patient_id, timestamp_utc DESC
        ),
        active_infections AS (
            SELECT DISTINCT patient_id
            FROM clinical_events
            WHERE record_type = 'event'
              AND payload->>'event_type' = 'infection_confirmed'
              AND timestamp_utc <= :as_of
              AND timestamp_utc >= :as_of - INTERVAL '72 hours'
        )
        SELECT COUNT(*) AS cnt
        FROM room_patients rp
        JOIN active_infections ai ON rp.patient_id = ai.patient_id
        WHERE rp.room_id = :room_id
          AND rp.event_type != 'discharge'
          AND rp.patient_id != :patient_id
    """
    colocated = await _scalar(conn, colocated_infections_sql, room_params)
    features["ward_colocated_active_infections"] = int(colocated) if colocated else 0
    features["ward_exposure_flag"] = features["ward_colocated_active_infections"] > 0

    # Staff contact count in last 24h (from treated_by / staff_contact events)
    staff_sql = """
        SELECT COUNT(DISTINCT payload->>'staff_id') AS staff_count
        FROM clinical_events
        WHERE patient_id = :patient_id
          AND record_type = 'event'
          AND payload->>'event_type' = 'staff_contact'
          AND timestamp_utc <= :as_of
          AND timestamp_utc >= :as_of - INTERVAL '24 hours'
    """
    staff_count = await _scalar(conn, staff_sql, params)
    features["ward_staff_contact_count_24h"] = int(staff_count) if staff_count else 0

    return features


# ---------------------------------------------------------------------------
# Composite: compute all features for a patient at a given as_of timestamp
# ---------------------------------------------------------------------------

async def compute_all_features(
    patient_id: str,
    as_of: datetime,
    conn: Conn,
) -> FeatureDict:
    """
    Compute and merge all feature groups for a patient at `as_of`.
    Returns a flat dict suitable for storage in the feature_store table.
    """
    vitals = await compute_vitals_features(patient_id, as_of, conn)
    labs = await compute_lab_features(patient_id, as_of, conn)
    chemo = await compute_chemo_features(patient_id, as_of, conn)
    infection = await compute_infection_history_features(patient_id, as_of, conn)
    ward = await compute_ward_context_features(patient_id, as_of, conn)

    return {
        **vitals,
        **labs,
        **chemo,
        **infection,
        **ward,
        "_computed_at": as_of.isoformat(),
        "_patient_id": patient_id,
    }
