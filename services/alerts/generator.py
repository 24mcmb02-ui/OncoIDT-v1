"""
OncoIDT Alert Service — Alert Generation and Threshold Evaluation.

Evaluates all trigger categories against incoming score update payloads and
feature snapshots, producing prioritised Alert objects.

Trigger categories (Requirement 10.1):
  - Infection risk threshold breach (default 0.6)
  - Deterioration risk threshold breach (default 0.65)
  - SIRS criteria met (≥ 2 criteria)
  - ANC critically low (< 0.5 × 10⁹/L)
  - NEWS2 escalation threshold (≥ 7)
  - Ward exposure event
  - Data source unavailability

Priority scoring (Requirement 10.2):
  priority = f(score_magnitude, score_delta_rate, patient_vulnerability_index)
  where vulnerability encodes ANC nadir severity, age, comorbidity burden.

Requirements: 10.1, 10.2
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Alert data model
# ---------------------------------------------------------------------------

AlertPriority = Literal["Critical", "High", "Medium", "Low"]
AlertType = Literal[
    "infection_risk",
    "deterioration_risk",
    "sirs_criteria",
    "anc_critical",
    "news2_escalation",
    "ward_exposure",
    "source_unavailability",
]


@dataclass
class Alert:
    alert_id: str
    patient_id: str | None          # None for ward-level / source alerts
    ward_id: str | None
    alert_type: AlertType
    priority: AlertPriority
    score: float | None             # triggering score value (if applicable)
    score_delta: float | None       # rate of change (if available)
    message: str
    details: dict[str, Any]
    generated_at: datetime
    escalation_count: int = 0
    acknowledged: bool = False
    snoozed_until: datetime | None = None
    top_features: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SIRS criteria evaluation
# ---------------------------------------------------------------------------

def _evaluate_sirs(features: dict[str, Any]) -> int:
    """
    Count how many SIRS criteria are met from the feature snapshot.
    Criteria:
      1. Temperature > 38°C or < 36°C
      2. Heart rate > 90 bpm
      3. Respiratory rate > 20 rpm
      4. WBC > 12 × 10⁹/L or < 4 × 10⁹/L
    Returns count of met criteria (0–4).
    """
    count = 0
    temp = features.get("temperature_c")
    if temp is not None and (temp > 38.0 or temp < 36.0):
        count += 1
    hr = features.get("heart_rate_bpm")
    if hr is not None and hr > 90:
        count += 1
    rr = features.get("respiratory_rate_rpm")
    if rr is not None and rr > 20:
        count += 1
    wbc = features.get("wbc")
    if wbc is not None and (wbc > 12.0 or wbc < 4.0):
        count += 1
    return count


def _compute_news2(features: dict[str, Any]) -> int:
    """
    Compute a simplified NEWS2 score from available feature values.
    Full NEWS2 uses: RR, SpO2, supplemental O2, temperature, SBP, HR, consciousness.
    Returns integer score (0–20).
    """
    score = 0

    rr = features.get("respiratory_rate_rpm")
    if rr is not None:
        if rr <= 8 or rr >= 25:
            score += 3
        elif rr >= 21:
            score += 2
        elif rr <= 11:
            score += 1

    spo2 = features.get("spo2_pct")
    if spo2 is not None:
        if spo2 <= 91:
            score += 3
        elif spo2 <= 93:
            score += 2
        elif spo2 <= 95:
            score += 1

    temp = features.get("temperature_c")
    if temp is not None:
        if temp <= 35.0 or temp >= 39.1:
            score += 3
        elif temp >= 38.1:
            score += 2
        elif temp <= 36.0:
            score += 1

    sbp = features.get("sbp_mmhg")
    if sbp is not None:
        if sbp <= 90 or sbp >= 220:
            score += 3
        elif sbp <= 100:
            score += 2
        elif sbp <= 110:
            score += 1

    hr = features.get("heart_rate_bpm")
    if hr is not None:
        if hr <= 40 or hr >= 131:
            score += 3
        elif hr >= 111:
            score += 2
        elif hr <= 50 or hr >= 91:
            score += 1

    gcs = features.get("gcs")
    if gcs is not None and gcs < 15:
        score += 3

    return score


# ---------------------------------------------------------------------------
# Vulnerability index
# ---------------------------------------------------------------------------

def _compute_vulnerability_index(features: dict[str, Any]) -> float:
    """
    Compute patient vulnerability index in [0, 1].
    Encodes ANC nadir severity, age, and comorbidity burden.
    """
    score = 0.0

    # ANC nadir severity (weight 0.5)
    anc = features.get("anc")
    if anc is not None:
        if anc < 0.1:
            score += 0.5
        elif anc < 0.5:
            score += 0.4
        elif anc < 1.0:
            score += 0.2
        elif anc < 2.0:
            score += 0.1

    # Age (weight 0.25) — older patients more vulnerable
    age = features.get("age_years")
    if age is not None:
        if age >= 75:
            score += 0.25
        elif age >= 65:
            score += 0.15
        elif age >= 55:
            score += 0.05

    # Comorbidity burden (weight 0.25) — immunosuppression score proxy
    immuno = features.get("immunosuppression_score")
    if immuno is not None:
        score += 0.25 * float(immuno)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

def _compute_priority(
    score_magnitude: float,
    score_delta_rate: float,
    vulnerability_index: float,
) -> AlertPriority:
    """
    priority = f(score_magnitude, score_delta_rate, patient_vulnerability_index)

    Composite priority score in [0, 1]:
      0.5 * score_magnitude + 0.3 * min(1, abs(score_delta_rate) * 10) + 0.2 * vulnerability_index

    Thresholds:
      >= 0.75 → Critical
      >= 0.55 → High
      >= 0.35 → Medium
      else    → Low
    """
    composite = (
        0.5 * score_magnitude
        + 0.3 * min(1.0, abs(score_delta_rate) * 10.0)
        + 0.2 * vulnerability_index
    )
    if composite >= 0.75:
        return "Critical"
    elif composite >= 0.55:
        return "High"
    elif composite >= 0.35:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Alert generators per trigger category
# ---------------------------------------------------------------------------

def _make_alert(
    alert_type: AlertType,
    patient_id: str | None,
    ward_id: str | None,
    priority: AlertPriority,
    score: float | None,
    score_delta: float | None,
    message: str,
    details: dict[str, Any],
    top_features: list[dict[str, Any]] | None = None,
) -> Alert:
    return Alert(
        alert_id=str(uuid.uuid4()),
        patient_id=patient_id,
        ward_id=ward_id,
        alert_type=alert_type,
        priority=priority,
        score=score,
        score_delta=score_delta,
        message=message,
        details=details,
        generated_at=datetime.now(timezone.utc),
        top_features=top_features or [],
    )


def evaluate_score_update(
    payload: dict[str, Any],
    features: dict[str, Any],
    previous_score: float | None = None,
) -> list[Alert]:
    """
    Evaluate a score update payload against all threshold trigger categories.

    Parameters
    ----------
    payload:
        Score update message from onco:inference:score_update stream.
    features:
        Current feature snapshot for the patient (may be empty dict).
    previous_score:
        Last known score for this patient/type/horizon (for delta rate).

    Returns
    -------
    List of Alert objects (may be empty if no thresholds breached).
    """
    alerts: list[Alert] = []
    patient_id: str = payload.get("patient_id", "")
    ward_id: str | None = payload.get("ward_id")
    score_type: str = payload.get("score_type", "")
    score: float = float(payload.get("final_score", payload.get("score", 0.0)))
    horizon: int = int(payload.get("forecast_horizon_hours", 24))
    top_features: list[dict[str, Any]] = payload.get("top_features", [])

    score_delta = (score - previous_score) if previous_score is not None else 0.0
    vulnerability = _compute_vulnerability_index(features)

    # ------------------------------------------------------------------
    # 1. Infection risk threshold
    # ------------------------------------------------------------------
    if score_type == "infection" and score >= settings.alert_infection_risk_threshold:
        priority = _compute_priority(score, score_delta, vulnerability)
        alerts.append(_make_alert(
            alert_type="infection_risk",
            patient_id=patient_id,
            ward_id=ward_id,
            priority=priority,
            score=score,
            score_delta=score_delta,
            message=(
                f"Infection risk score {score:.2f} at {horizon}h horizon "
                f"exceeds threshold {settings.alert_infection_risk_threshold:.2f}"
            ),
            details={
                "score_type": score_type,
                "forecast_horizon_hours": horizon,
                "threshold": settings.alert_infection_risk_threshold,
                "model_version": payload.get("model_version"),
                "feature_snapshot_id": payload.get("feature_snapshot_id"),
            },
            top_features=top_features,
        ))

    # ------------------------------------------------------------------
    # 2. Deterioration risk threshold
    # ------------------------------------------------------------------
    if score_type == "deterioration" and score >= settings.alert_deterioration_risk_threshold:
        priority = _compute_priority(score, score_delta, vulnerability)
        alerts.append(_make_alert(
            alert_type="deterioration_risk",
            patient_id=patient_id,
            ward_id=ward_id,
            priority=priority,
            score=score,
            score_delta=score_delta,
            message=(
                f"Deterioration risk score {score:.2f} at {horizon}h horizon "
                f"exceeds threshold {settings.alert_deterioration_risk_threshold:.2f}"
            ),
            details={
                "score_type": score_type,
                "forecast_horizon_hours": horizon,
                "threshold": settings.alert_deterioration_risk_threshold,
                "model_version": payload.get("model_version"),
                "top_features": top_features[:3] if top_features else [],
            },
            top_features=top_features[:3],
        ))

    # ------------------------------------------------------------------
    # 3. SIRS criteria
    # ------------------------------------------------------------------
    if features:
        sirs_count = _evaluate_sirs(features)
        if sirs_count >= 2:
            priority = _compute_priority(sirs_count / 4.0, 0.0, vulnerability)
            alerts.append(_make_alert(
                alert_type="sirs_criteria",
                patient_id=patient_id,
                ward_id=ward_id,
                priority=priority,
                score=float(sirs_count),
                score_delta=None,
                message=f"Patient meets {sirs_count}/4 SIRS criteria — infection/sepsis risk elevated",
                details={
                    "sirs_count": sirs_count,
                    "temperature_c": features.get("temperature_c"),
                    "heart_rate_bpm": features.get("heart_rate_bpm"),
                    "respiratory_rate_rpm": features.get("respiratory_rate_rpm"),
                    "wbc": features.get("wbc"),
                },
            ))

    # ------------------------------------------------------------------
    # 4. ANC critically low
    # ------------------------------------------------------------------
    if features:
        anc = features.get("anc")
        if anc is not None and anc < settings.alert_anc_critical_threshold:
            priority = _compute_priority(
                1.0 - (anc / settings.alert_anc_critical_threshold),
                0.0,
                vulnerability,
            )
            alerts.append(_make_alert(
                alert_type="anc_critical",
                patient_id=patient_id,
                ward_id=ward_id,
                priority=priority,
                score=anc,
                score_delta=None,
                message=(
                    f"ANC critically low: {anc:.2f} × 10⁹/L "
                    f"(threshold < {settings.alert_anc_critical_threshold:.1f})"
                ),
                details={
                    "anc": anc,
                    "threshold": settings.alert_anc_critical_threshold,
                    "immunosuppression_score": features.get("immunosuppression_score"),
                    "chemo_cycle_phase": features.get("chemo_cycle_phase"),
                },
            ))

    # ------------------------------------------------------------------
    # 5. NEWS2 escalation
    # ------------------------------------------------------------------
    if features:
        news2 = _compute_news2(features)
        if news2 >= 7:
            priority = _compute_priority(min(1.0, news2 / 20.0), 0.0, vulnerability)
            alerts.append(_make_alert(
                alert_type="news2_escalation",
                patient_id=patient_id,
                ward_id=ward_id,
                priority=priority,
                score=float(news2),
                score_delta=None,
                message=f"NEWS2 score {news2} ≥ 7 — urgent clinical review required",
                details={
                    "news2_score": news2,
                    "respiratory_rate_rpm": features.get("respiratory_rate_rpm"),
                    "spo2_pct": features.get("spo2_pct"),
                    "temperature_c": features.get("temperature_c"),
                    "sbp_mmhg": features.get("sbp_mmhg"),
                    "heart_rate_bpm": features.get("heart_rate_bpm"),
                    "gcs": features.get("gcs"),
                },
            ))

    return alerts


def generate_ward_exposure_alert(
    ward_id: str,
    source_patient_id: str,
    affected_patient_ids: list[str],
    pathogen: str | None,
    confidence: float,
) -> Alert:
    """Generate a ward-level exposure event alert (Requirement 10.1)."""
    priority: AlertPriority = "High" if confidence >= 0.7 else "Medium"
    return _make_alert(
        alert_type="ward_exposure",
        patient_id=source_patient_id,
        ward_id=ward_id,
        priority=priority,
        score=confidence,
        score_delta=None,
        message=(
            f"Ward exposure event: {len(affected_patient_ids)} patient(s) potentially exposed"
            + (f" to {pathogen}" if pathogen else "")
        ),
        details={
            "source_patient_id": source_patient_id,
            "affected_patient_ids": affected_patient_ids,
            "pathogen": pathogen,
            "confidence": confidence,
        },
    )


def generate_source_unavailability_alert(
    source_system: str,
    silent_minutes: float,
    ward_id: str | None = None,
) -> Alert:
    """Generate a data source unavailability alert (Requirement 10.1)."""
    return _make_alert(
        alert_type="source_unavailability",
        patient_id=None,
        ward_id=ward_id,
        priority="High",
        score=None,
        score_delta=None,
        message=f"Data source '{source_system}' has been silent for {silent_minutes:.0f} minutes",
        details={
            "source_system": source_system,
            "silent_minutes": silent_minutes,
        },
    )
