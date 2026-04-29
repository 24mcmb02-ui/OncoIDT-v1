"""
OncoIDT Hybrid Reasoner — Score Fusion Logic.

Fuses ML risk scores with clinical rule overrides according to fusion_mode:
  - "hybrid"    : apply hard floors, blend soft rule deltas with ML score
  - "rule_only" : ignore ML score, use only rule-derived floors/deltas
  - "ml_only"   : ignore all rules, pass ML score through unchanged

Annotates output with full provenance and logs to audit log.

Requirements: 8.2, 8.4, 8.5, 8.7
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.audit import append_audit_entry
from shared.models import RuleOverride

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class FusionMode(str, Enum):
    HYBRID = "hybrid"
    RULE_ONLY = "rule_only"
    ML_ONLY = "ml_only"


@dataclass
class FusionResult:
    patient_id: str
    score_type: str                     # "infection" | "deterioration"
    forecast_horizon_hours: int
    ml_score: float
    final_score: float
    uncertainty_lower: float
    uncertainty_upper: float
    rule_overrides: list[RuleOverride]
    fusion_mode: str
    model_version: str
    feature_snapshot_id: str
    staleness_flag: bool
    timestamp: datetime


# ---------------------------------------------------------------------------
# Core fusion function
# ---------------------------------------------------------------------------

def fuse_scores(
    score_payload: dict[str, Any],
    rule_overrides: list[RuleOverride],
    fusion_mode: FusionMode,
    soft_rule_weight: float = 0.3,
) -> FusionResult:
    """
    Fuse an ML risk score with clinical rule overrides.

    Parameters
    ----------
    score_payload:
        Score update message from the inference service.
    rule_overrides:
        Triggered RuleOverride objects from the rule engine.
        Callers are responsible for pre-filtering to relevant rules.
    fusion_mode:
        One of FusionMode.HYBRID, FusionMode.RULE_ONLY, FusionMode.ML_ONLY.
    soft_rule_weight:
        Weight applied to the sum of soft-rule deltas in hybrid mode (default 0.3).

    Returns
    -------
    FusionResult with full provenance annotation.
    """
    ml_score: float = float(score_payload["score"])
    score_type: str = score_payload["score_type"]

    hard_rules = [r for r in rule_overrides if r.rule_id.startswith("hard_")]
    soft_rules = [r for r in rule_overrides if not r.rule_id.startswith("hard_")]

    max_hard_floor = max((r.score_floor for r in hard_rules), default=0.0)
    soft_delta_sum = sum(r.score_floor for r in soft_rules)

    if fusion_mode is FusionMode.ML_ONLY:
        final_score = ml_score
        applied_overrides: list[RuleOverride] = []

    elif fusion_mode is FusionMode.RULE_ONLY:
        final_score = max_hard_floor + soft_delta_sum
        final_score = max(0.0, min(1.0, final_score))
        applied_overrides = list(rule_overrides)

    else:  # HYBRID (default)
        final_score = max(ml_score, max_hard_floor)
        final_score = final_score + soft_rule_weight * soft_delta_sum
        final_score = max(0.0, min(1.0, final_score))
        applied_overrides = list(rule_overrides)

    ts_raw = score_payload.get("timestamp")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    return FusionResult(
        patient_id=score_payload["patient_id"],
        score_type=score_type,
        forecast_horizon_hours=int(score_payload["forecast_horizon_hours"]),
        ml_score=ml_score,
        final_score=final_score,
        uncertainty_lower=float(score_payload["uncertainty_lower"]),
        uncertainty_upper=float(score_payload["uncertainty_upper"]),
        rule_overrides=applied_overrides,
        fusion_mode=fusion_mode.value,
        model_version=score_payload["model_version"],
        feature_snapshot_id=score_payload["feature_snapshot_id"],
        staleness_flag=bool(score_payload.get("staleness_flag", False)),
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

async def log_fusion_decision(
    session: AsyncSession,
    patient_id: str,
    result: FusionResult,
) -> None:
    """Append a fusion decision entry to the audit log."""
    details: dict[str, Any] = {
        "ml_score": result.ml_score,
        "final_score": result.final_score,
        "fusion_mode": result.fusion_mode,
        "rule_override_ids": [r.rule_id for r in result.rule_overrides],
        "score_type": result.score_type,
        "horizon": result.forecast_horizon_hours,
    }
    await append_audit_entry(
        session,
        user_id="reasoner-service",
        action="fusion_decision",
        resource_type="risk_score",
        resource_id=patient_id,
        details=details,
    )
    logger.debug(
        "Fusion decision logged: patient=%s mode=%s ml=%.3f final=%.3f",
        patient_id,
        result.fusion_mode,
        result.ml_score,
        result.final_score,
    )
