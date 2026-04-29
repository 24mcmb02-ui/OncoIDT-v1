"""
OncoIDT Hybrid Reasoner — Clinical Rule Engine.

Loads clinical rules from rules.yaml with hot-reload via watchfiles.
Evaluates hard rules (non-configurable) and soft rules (YAML-configurable)
against a patient feature snapshot.

Hard rules (always enforced):
  - SIRS >= 2 criteria → infection_risk >= 0.7
  - NEWS2 >= 7 → deterioration_risk >= 0.8
  - ANC < 0.5 × 10⁹/L + temp > 38.3°C → infection_risk >= 0.85

Requirements: 8.1, 8.2, 4.4, 5.6
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import yaml

from shared.models import RuleOverride

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft rule config model
# ---------------------------------------------------------------------------

@dataclass
class SoftRuleConfig:
    rule_id: str
    description: str
    target_score: str          # "infection_risk" | "deterioration_risk"
    condition_field: str
    condition_operator: str    # ">" | "<" | ">=" | "<=" | "=="
    condition_threshold: float
    score_delta: float
    enabled: bool = True


# ---------------------------------------------------------------------------
# NEWS2 scoring helpers
# ---------------------------------------------------------------------------

def _news2_resp_rate(rr: float) -> int:
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def _news2_spo2(spo2: float) -> int:
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def _news2_sbp(sbp: float) -> int:
    if sbp <= 90:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    if sbp <= 219:
        return 0
    return 3


def _news2_hr(hr: float) -> int:
    if hr <= 40:
        return 3
    if hr <= 50:
        return 1
    if hr <= 90:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3


def _news2_temp(temp: float) -> int:
    if temp <= 35.0:
        return 3
    if temp <= 36.0:
        return 1
    if temp <= 38.0:
        return 0
    if temp <= 39.0:
        return 1
    return 2


def _news2_gcs(gcs: int) -> int:
    return 0 if gcs == 15 else 3


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

_OPERATORS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


class RuleEngine:
    """Evaluates hard and soft clinical rules against a patient feature snapshot."""

    def __init__(self, rules_yaml_path: str) -> None:
        self._path = rules_yaml_path
        self._soft_rules: list[SoftRuleConfig] = []
        self._lock = threading.RLock()
        self._reload_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_rules(self, feature_snapshot: dict) -> list[RuleOverride]:
        """Evaluate all hard and soft rules; return triggered RuleOverride list."""
        overrides: list[RuleOverride] = []

        overrides.extend(self._evaluate_hard_rules(feature_snapshot))
        overrides.extend(self._evaluate_soft_rules(feature_snapshot))

        return overrides

    def get_active_rules(self) -> list[dict]:
        """Return list of active soft rule configs as plain dicts (for API)."""
        with self._lock:
            return [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "target_score": r.target_score,
                    "condition_field": r.condition_field,
                    "condition_operator": r.condition_operator,
                    "condition_threshold": r.condition_threshold,
                    "score_delta": r.score_delta,
                    "enabled": r.enabled,
                }
                for r in self._soft_rules
            ]

    def update_soft_rule(self, rule_id: str, updates: dict) -> None:
        """Update a soft rule in memory and persist the change to YAML."""
        with self._lock:
            for rule in self._soft_rules:
                if rule.rule_id == rule_id:
                    for k, v in updates.items():
                        if hasattr(rule, k):
                            setattr(rule, k, v)
                    self._persist_rules()
                    return
        raise KeyError(f"Soft rule '{rule_id}' not found")

    # ------------------------------------------------------------------
    # Hard rules
    # ------------------------------------------------------------------

    def _evaluate_hard_rules(self, features: dict) -> list[RuleOverride]:
        overrides: list[RuleOverride] = []

        # Hard rule 1: SIRS >= 2 → infection_risk floor >= 0.7
        sirs_count = self._compute_sirs_count(features)
        if sirs_count >= 2:
            overrides.append(RuleOverride(
                rule_id="hard_sirs_infection_risk",
                threshold_value=2.0,
                triggered_value=float(sirs_count),
                score_floor=0.7,
            ))

        # Hard rule 2: NEWS2 >= 7 → deterioration_risk floor >= 0.8
        news2 = self._compute_news2(features)
        if news2 >= 7:
            overrides.append(RuleOverride(
                rule_id="hard_news2_deterioration_risk",
                threshold_value=7.0,
                triggered_value=news2,
                score_floor=0.8,
            ))

        # Hard rule 3: ANC < 0.5 AND temp > 38.3 → infection_risk floor >= 0.85
        anc = features.get("anc")
        temp = features.get("temperature_c")
        if anc is not None and temp is not None and anc < 0.5 and temp > 38.3:
            overrides.append(RuleOverride(
                rule_id="hard_anc_fever_infection_risk",
                threshold_value=0.5,
                triggered_value=anc,
                score_floor=0.85,
            ))

        return overrides

    # ------------------------------------------------------------------
    # Soft rules
    # ------------------------------------------------------------------

    def _evaluate_soft_rules(self, features: dict) -> list[RuleOverride]:
        overrides: list[RuleOverride] = []
        with self._lock:
            rules = list(self._soft_rules)

        for rule in rules:
            if not rule.enabled:
                continue
            value = features.get(rule.condition_field)
            if value is None:
                continue
            op = _OPERATORS.get(rule.condition_operator)
            if op is None:
                logger.warning("Unknown operator '%s' in rule '%s'", rule.condition_operator, rule.rule_id)
                continue
            if op(float(value), rule.condition_threshold):
                overrides.append(RuleOverride(
                    rule_id=rule.rule_id,
                    threshold_value=rule.condition_threshold,
                    triggered_value=float(value),
                    score_floor=rule.score_delta,  # delta used as floor contribution
                ))

        return overrides

    # ------------------------------------------------------------------
    # SIRS / NEWS2 helpers
    # ------------------------------------------------------------------

    def _compute_sirs_count(self, features: dict) -> int:
        """Count how many SIRS criteria are met (0–4)."""
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

    def _compute_news2(self, features: dict) -> float:
        """Return NEWS2 score — use pre-computed value if present, else derive from vitals."""
        if "news2_score" in features and features["news2_score"] is not None:
            return float(features["news2_score"])

        score = 0
        rr = features.get("respiratory_rate_rpm")
        if rr is not None:
            score += _news2_resp_rate(float(rr))
        spo2 = features.get("spo2_pct")
        if spo2 is not None:
            score += _news2_spo2(float(spo2))
        sbp = features.get("sbp_mmhg")
        if sbp is not None:
            score += _news2_sbp(float(sbp))
        hr = features.get("heart_rate_bpm")
        if hr is not None:
            score += _news2_hr(float(hr))
        temp = features.get("temperature_c")
        if temp is not None:
            score += _news2_temp(float(temp))
        gcs = features.get("gcs")
        if gcs is not None:
            score += _news2_gcs(int(gcs))

        return float(score)

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    def _reload_rules(self) -> None:
        """Load (or reload) soft rules from the YAML file."""
        if not os.path.exists(self._path):
            logger.warning("rules.yaml not found at %s — no soft rules loaded", self._path)
            with self._lock:
                self._soft_rules = []
            return

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            raw_rules = data.get("soft_rules", [])
            parsed = [
                SoftRuleConfig(
                    rule_id=r["rule_id"],
                    description=r.get("description", ""),
                    target_score=r["target_score"],
                    condition_field=r["condition_field"],
                    condition_operator=r["condition_operator"],
                    condition_threshold=float(r["condition_threshold"]),
                    score_delta=float(r["score_delta"]),
                    enabled=bool(r.get("enabled", True)),
                )
                for r in raw_rules
            ]
            with self._lock:
                self._soft_rules = parsed
            logger.info("Loaded %d soft rules from %s", len(parsed), self._path)
        except Exception:
            logger.exception("Failed to reload rules from %s", self._path)



    def _persist_rules(self) -> None:
        """Write current soft rules back to YAML (called under lock)."""
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            data["soft_rules"] = [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "target_score": r.target_score,
                    "condition_field": r.condition_field,
                    "condition_operator": r.condition_operator,
                    "condition_threshold": r.condition_threshold,
                    "score_delta": r.score_delta,
                    "enabled": r.enabled,
                }
                for r in self._soft_rules
            ]
            with open(self._path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True)
        except Exception:
            logger.exception("Failed to persist rules to %s", self._path)


# ---------------------------------------------------------------------------
# Background file watcher
# ---------------------------------------------------------------------------

async def watch_rules_file(path: str, engine: RuleEngine) -> None:
    """Async background task: reload rules whenever rules.yaml changes."""
    try:
        from watchfiles import awatch
        async for _ in awatch(path):
            logger.info("rules.yaml changed — reloading soft rules")
            engine._reload_rules()
    except Exception:
        logger.exception("watch_rules_file terminated unexpectedly")


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_engine_instance: RuleEngine | None = None
_engine_lock = threading.Lock()

_DEFAULT_YAML_PATH = os.path.join(os.path.dirname(__file__), "rules.yaml")


def get_rule_engine(rules_yaml_path: str = _DEFAULT_YAML_PATH) -> RuleEngine:
    """Return the module-level RuleEngine singleton, creating it if needed."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = RuleEngine(rules_yaml_path)
    return _engine_instance
