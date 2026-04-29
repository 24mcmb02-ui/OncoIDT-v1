"""
OncoIDT Explainability Service — Natural Language Renderer.

Translates SHAP feature attributions into clinically meaningful natural
language sentences for display in the Clinician Dashboard.

Template patterns:
  Per-feature:
    "ANC of {value} × 10⁹/L is {severity_label} and is the {rank} driver
     of this {score_type} score"

  Delta (simulation):
    "Administering {intervention} is predicted to reduce {score_type} by
     {delta:.0%} primarily by reducing the contribution of [{top_feature}]"

Requirements: 11.2, 11.4, 11.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from services.explainability.shap_engine import (
    DeltaExplanation,
    ExplanationResult,
    FeatureAttribution,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical vocabulary mappings
# ---------------------------------------------------------------------------

# Human-readable display names for each feature
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "anc": "ANC",
    "wbc": "WBC",
    "lymphocytes": "lymphocyte count",
    "crp_mg_l": "CRP",
    "procalcitonin_ug_l": "procalcitonin",
    "temperature_c": "temperature",
    "heart_rate_bpm": "heart rate",
    "respiratory_rate_rpm": "respiratory rate",
    "sbp_mmhg": "systolic blood pressure",
    "dbp_mmhg": "diastolic blood pressure",
    "spo2_pct": "oxygen saturation",
    "gcs": "GCS",
    "days_since_last_chemo_dose": "days since last chemotherapy dose",
    "immunosuppression_score": "immunosuppression score",
    "chemo_cycle_number": "chemotherapy cycle number",
    "vitals_mean_1h": "mean vitals over 1 hour",
    "vitals_std_1h": "vitals variability over 1 hour",
    "vitals_min_1h": "minimum vitals over 1 hour",
    "vitals_max_1h": "maximum vitals over 1 hour",
    "vitals_mean_6h": "mean vitals over 6 hours",
    "vitals_std_6h": "vitals variability over 6 hours",
    "vitals_min_6h": "minimum vitals over 6 hours",
    "vitals_max_6h": "maximum vitals over 6 hours",
    "vitals_mean_24h": "mean vitals over 24 hours",
    "vitals_std_24h": "vitals variability over 24 hours",
    "vitals_min_24h": "minimum vitals over 24 hours",
    "vitals_max_24h": "maximum vitals over 24 hours",
    "anc_slope_6h": "ANC trend over 6 hours",
    "anc_slope_24h": "ANC trend over 24 hours",
    "time_since_last_antibiotic_hours": "time since last antibiotic dose",
    "prior_infection_count": "prior infection count",
    "antibiotic_active": "active antibiotic therapy",
    "co_located_active_infections": "co-located patients with active infections",
    "staff_contact_count_24h": "staff contacts in the last 24 hours",
}

# Units for features that have them
FEATURE_UNITS: dict[str, str] = {
    "anc": "× 10⁹/L",
    "wbc": "× 10⁹/L",
    "lymphocytes": "× 10⁹/L",
    "crp_mg_l": "mg/L",
    "procalcitonin_ug_l": "μg/L",
    "temperature_c": "°C",
    "heart_rate_bpm": "bpm",
    "respiratory_rate_rpm": "breaths/min",
    "sbp_mmhg": "mmHg",
    "dbp_mmhg": "mmHg",
    "spo2_pct": "%",
    "days_since_last_chemo_dose": "days",
    "time_since_last_antibiotic_hours": "hours",
    "anc_slope_6h": "× 10⁹/L per hour",
    "anc_slope_24h": "× 10⁹/L per hour",
}

# Severity thresholds for key clinical features (value → label)
# Each entry is a list of (threshold, label) tuples in ascending order.
# The label for the highest threshold that the value exceeds is used.
_SEVERITY_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "anc": [
        (0.0, "critically low"),
        (0.5, "severely low"),
        (1.0, "low"),
        (1.5, "borderline low"),
        (2.0, "normal"),
    ],
    "temperature_c": [
        (0.0, "hypothermic"),
        (36.0, "normal"),
        (37.5, "mildly elevated"),
        (38.0, "febrile"),
        (39.0, "high fever"),
    ],
    "heart_rate_bpm": [
        (0.0, "bradycardic"),
        (60.0, "normal"),
        (100.0, "tachycardic"),
        (120.0, "significantly elevated"),
    ],
    "respiratory_rate_rpm": [
        (0.0, "bradypnoeic"),
        (12.0, "normal"),
        (20.0, "mildly elevated"),
        (25.0, "elevated"),
    ],
    "spo2_pct": [
        (0.0, "critically low"),
        (88.0, "severely low"),
        (92.0, "low"),
        (95.0, "borderline"),
        (96.0, "normal"),
    ],
    "crp_mg_l": [
        (0.0, "normal"),
        (10.0, "mildly elevated"),
        (50.0, "elevated"),
        (100.0, "significantly elevated"),
    ],
    "procalcitonin_ug_l": [
        (0.0, "normal"),
        (0.5, "borderline"),
        (2.0, "elevated"),
        (10.0, "significantly elevated"),
    ],
    "immunosuppression_score": [
        (0.0, "minimal"),
        (0.3, "mild"),
        (0.6, "moderate"),
        (0.8, "severe"),
    ],
}

# Ordinal rank labels
_RANK_LABELS: dict[int, str] = {
    1: "primary",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
}

# Score type display names
_SCORE_TYPE_LABELS: dict[str, str] = {
    "infection": "infection risk",
    "deterioration": "deterioration risk",
}


def _get_severity_label(feature_name: str, value: float) -> str:
    """Return a clinical severity label for a feature value."""
    thresholds = _SEVERITY_THRESHOLDS.get(feature_name)
    if not thresholds:
        return "notable"

    label = thresholds[0][1]
    for threshold, lbl in thresholds:
        if value >= threshold:
            label = lbl
    return label


def _format_value(feature_name: str, value: float | None) -> str:
    """Format a feature value with its unit."""
    if value is None:
        return "unknown"
    unit = FEATURE_UNITS.get(feature_name, "")
    if feature_name == "antibiotic_active":
        return "active" if value >= 0.5 else "inactive"
    if unit:
        return f"{value:.2g} {unit}"
    return f"{value:.2g}"


def _direction_phrase(shap_value: float) -> str:
    """Return 'increasing' or 'reducing' based on SHAP sign."""
    return "increasing" if shap_value > 0 else "reducing"


# ---------------------------------------------------------------------------
# Per-feature sentence renderer
# ---------------------------------------------------------------------------

def render_feature_sentence(
    attr: FeatureAttribution,
    score_type: str,
    is_rule_driven: bool = False,
) -> str:
    """
    Render a single feature attribution as a clinical sentence.

    Examples:
      "ANC of 0.3 × 10⁹/L is critically low and is the primary driver of
       this infection risk score"
      "Temperature of 38.5 °C is febrile and is the second driver of this
       infection risk score (rule-triggered)"

    Requirements: 11.2, 11.4
    """
    display_name = FEATURE_DISPLAY_NAMES.get(attr.feature_name, attr.feature_name)
    score_label = _SCORE_TYPE_LABELS.get(score_type, score_type)
    rank_label = _RANK_LABELS.get(attr.rank, f"{attr.rank}th")

    if attr.feature_value is not None:
        value_str = _format_value(attr.feature_name, attr.feature_value)
        severity = _get_severity_label(attr.feature_name, attr.feature_value)
        sentence = (
            f"{display_name} of {value_str} is {severity} and is the "
            f"{rank_label} driver of this {score_label} score"
        )
    else:
        direction = _direction_phrase(attr.shap_value)
        sentence = (
            f"{display_name} is {direction} this {score_label} score "
            f"(ranked {rank_label})"
        )

    if is_rule_driven:
        sentence += " (rule-triggered)"

    return sentence


# ---------------------------------------------------------------------------
# Full explanation renderer
# ---------------------------------------------------------------------------

@dataclass
class RenderedExplanation:
    """Human-readable explanation for a risk score."""
    patient_id: str
    score_type: str
    forecast_horizon_hours: int
    summary: str
    feature_sentences: list[str]
    rule_sentences: list[str]
    is_rule_driven: bool


def render_explanation(result: ExplanationResult) -> RenderedExplanation:
    """
    Render a full ExplanationResult into natural language.

    Distinguishes rule-triggered vs ML-driven contributions (Requirement 11.4).

    Returns:
        RenderedExplanation with a summary sentence and per-feature sentences.
    """
    score_label = _SCORE_TYPE_LABELS.get(result.score_type, result.score_type)
    horizon_label = f"{result.forecast_horizon_hours}h"

    feature_sentences: list[str] = []
    for attr in result.top_features:
        sentence = render_feature_sentence(attr, result.score_type, is_rule_driven=False)
        feature_sentences.append(sentence)

    rule_sentences: list[str] = []
    if result.is_rule_driven and result.rule_ids:
        for rule_id in result.rule_ids:
            rule_sentences.append(
                f"Clinical rule '{rule_id}' has overridden the ML score for "
                f"this {score_label} assessment"
            )

    # Build summary
    if result.top_features:
        top = result.top_features[0]
        top_name = FEATURE_DISPLAY_NAMES.get(top.feature_name, top.feature_name)
        if result.is_rule_driven:
            summary = (
                f"This {score_label} at {horizon_label} is primarily driven by "
                f"{top_name} and is subject to clinical rule overrides."
            )
        else:
            summary = (
                f"This {score_label} at {horizon_label} is primarily driven by "
                f"{top_name}."
            )
    else:
        summary = f"No feature attribution available for this {score_label} at {horizon_label}."

    return RenderedExplanation(
        patient_id=result.patient_id,
        score_type=result.score_type,
        forecast_horizon_hours=result.forecast_horizon_hours,
        summary=summary,
        feature_sentences=feature_sentences,
        rule_sentences=rule_sentences,
        is_rule_driven=result.is_rule_driven,
    )


# ---------------------------------------------------------------------------
# Delta explanation renderer (simulation counterfactuals)
# ---------------------------------------------------------------------------

@dataclass
class RenderedDeltaExplanation:
    """Human-readable delta explanation for a counterfactual simulation."""
    patient_id: str
    score_type: str
    summary: str
    feature_sentences: list[str]


def render_delta_explanation(
    delta: DeltaExplanation,
    intervention_label: str = "the intervention",
) -> RenderedDeltaExplanation:
    """
    Render a DeltaExplanation into natural language for simulation results.

    Template:
      "Administering {intervention} is predicted to reduce {score_type} by
       {delta:.0%} primarily by reducing the contribution of [{top_feature}]"

    Requirements: 11.5
    """
    score_label = _SCORE_TYPE_LABELS.get(delta.score_type, delta.score_type)
    delta_pct = abs(delta.score_delta)
    direction = "reduce" if delta.score_delta < 0 else "increase"

    feature_sentences: list[str] = []
    top_feature_names: list[str] = []

    for attr in delta.top_delta_features:
        display_name = FEATURE_DISPLAY_NAMES.get(attr.feature_name, attr.feature_name)
        top_feature_names.append(display_name)
        direction_word = _direction_phrase(attr.shap_value)
        sentence = (
            f"{display_name} contribution {direction_word} by "
            f"{abs(attr.shap_value):.3f} SHAP units"
        )
        feature_sentences.append(sentence)

    if top_feature_names:
        top_list = ", ".join(top_feature_names[:2])
        summary = (
            f"Administering {intervention_label} is predicted to {direction} "
            f"{score_label} by {delta_pct:.0%} primarily by {direction}ing "
            f"the contribution of [{top_list}]"
        )
    else:
        summary = (
            f"Administering {intervention_label} is predicted to {direction} "
            f"{score_label} by {delta_pct:.0%}"
        )

    return RenderedDeltaExplanation(
        patient_id=delta.patient_id,
        score_type=delta.score_type,
        summary=summary,
        feature_sentences=feature_sentences,
    )
