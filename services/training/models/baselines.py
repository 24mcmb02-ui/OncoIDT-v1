"""
Baseline models for OncoIDT ablation suite.

- NEWS2Scorer:     Deterministic rule-based scorer from a vitals snapshot.
- LSTMBaseline:    Standard LSTM on mean-imputed time series with the same
                   task heads as NeuralCDE.
- XGBoostBaseline: Gradient boosting on hand-crafted features from the
                   feature store snapshot.

All baselines expose the same predict() interface so the training pipeline
can swap them in without code changes.

Requirements: 6.6, 14.6
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NEWS2 Scorer (deterministic, no ML)
# ---------------------------------------------------------------------------

@dataclass
class VitalsInput:
    """Minimal vitals snapshot for NEWS2 scoring."""
    respiratory_rate: float | None = None   # breaths/min
    spo2: float | None = None               # %
    on_supplemental_o2: bool = False
    temperature_c: float | None = None
    sbp_mmhg: float | None = None
    heart_rate_bpm: float | None = None
    consciousness: str = "A"                # A=Alert, C=Confusion, V=Voice, P=Pain, U=Unresponsive


def _score_rr(rr: float | None) -> int:
    if rr is None:
        return 0
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def _score_spo2(spo2: float | None, on_o2: bool) -> int:
    if spo2 is None:
        return 0
    if on_o2:
        if spo2 >= 97:
            return 3
        if spo2 >= 95:
            return 2
        if spo2 >= 93:
            return 1
        return 0
    else:
        if spo2 >= 96:
            return 0
        if spo2 >= 94:
            return 1
        if spo2 >= 92:
            return 2
        return 3


def _score_temp(temp: float | None) -> int:
    if temp is None:
        return 0
    if temp <= 35.0:
        return 3
    if temp <= 36.0:
        return 1
    if temp <= 38.0:
        return 0
    if temp <= 39.0:
        return 1
    return 2


def _score_sbp(sbp: float | None) -> int:
    if sbp is None:
        return 0
    if sbp <= 90:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    if sbp <= 219:
        return 0
    return 3


def _score_hr(hr: float | None) -> int:
    if hr is None:
        return 0
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


def _score_consciousness(level: str) -> int:
    return 0 if level == "A" else 3


class NEWS2Scorer:
    """Deterministic NEWS2 rule-based scorer.

    Computes the National Early Warning Score 2 from a vitals snapshot and
    maps it to a deterioration risk probability in [0, 1].

    The mapping is a simple piecewise linear function calibrated to the
    NEWS2 clinical risk bands:
        0–4   → Low risk
        5–6   → Medium risk
        ≥7    → High risk (triggers hard override in Hybrid Reasoner)
    """

    # NEWS2 max possible score is 20; normalise to [0, 1]
    _MAX_SCORE = 20.0

    def score(self, vitals: VitalsInput) -> int:
        """Compute raw integer NEWS2 score (0–20)."""
        return (
            _score_rr(vitals.respiratory_rate)
            + _score_spo2(vitals.spo2, vitals.on_supplemental_o2)
            + (2 if vitals.on_supplemental_o2 else 0)
            + _score_temp(vitals.temperature_c)
            + _score_sbp(vitals.sbp_mmhg)
            + _score_hr(vitals.heart_rate_bpm)
            + _score_consciousness(vitals.consciousness)
        )

    def predict(self, vitals: VitalsInput) -> dict[str, float]:
        """Return NEWS2 score and a normalised deterioration risk probability.

        Returns:
            Dict with 'news2_score' (int) and 'deterioration_risk' (float in [0,1]).
        """
        raw = self.score(vitals)
        risk = min(raw / self._MAX_SCORE, 1.0)
        return {"news2_score": raw, "deterioration_risk": risk}


# ---------------------------------------------------------------------------
# LSTM Baseline
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """Standard LSTM on mean-imputed time series.

    Uses the same task heads as NeuralCDE (InfectionHead, DeteriorationHead,
    SurvivalHead) for a fair ablation comparison.

    Args:
        input_dim:   Number of clinical features per time step.
        hidden_dim:  LSTM hidden state dimension (default 256).
        num_layers:  Number of stacked LSTM layers (default 2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Shared representation (mirrors NeuralCDE)
        self.shared_repr = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Task heads (same architecture as NeuralCDE)
        self.infection_head = nn.Linear(256, 4)
        self.deterioration_head = nn.Linear(256, 3)
        self.survival_head = nn.Linear(256, 2)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       Mean-imputed time series, shape (batch, seq_len, input_dim).
                     NaN values should be replaced with feature means before
                     calling this method.
            lengths: Optional sequence lengths for packing, shape (batch,).

        Returns:
            infection_risk:    (batch, 4)
            deterioration_risk:(batch, 3)
            survival_params:   (batch, 2) — Weibull (shape, scale) via softplus
            repr:              (batch, 256)
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # Take the last layer's hidden state
        h_last = h_n[-1]   # (batch, hidden_dim)

        repr_vec = self.shared_repr(h_last)

        infection_risk = torch.sigmoid(self.infection_head(repr_vec))
        deterioration_risk = torch.sigmoid(self.deterioration_head(repr_vec))
        survival_params = nn.functional.softplus(self.survival_head(repr_vec))

        return infection_risk, deterioration_risk, survival_params, repr_vec


# ---------------------------------------------------------------------------
# XGBoost Baseline
# ---------------------------------------------------------------------------

class XGBoostBaseline:
    """Gradient boosting on hand-crafted features from the feature store.

    Wraps XGBoost classifiers for infection and deterioration prediction.
    One model per forecast horizon is trained independently.

    Infection horizons:    [6, 12, 24, 48] hours
    Deterioration horizons:[6, 12, 24] hours

    Args:
        xgb_params: XGBoost hyperparameters passed to XGBClassifier.
                    Defaults to sensible oncology-task settings.
    """

    _DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 5,   # handles class imbalance (~5-15% event rate)
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42,
    }

    def __init__(self, xgb_params: dict[str, Any] | None = None) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostBaseline. "
                "Install with: pip install xgboost"
            ) from exc

        params = {**self._DEFAULT_PARAMS, **(xgb_params or {})}

        from xgboost import XGBClassifier  # noqa: F811

        # One classifier per horizon
        self.infection_models: dict[int, Any] = {
            h: XGBClassifier(**params) for h in [6, 12, 24, 48]
        }
        self.deterioration_models: dict[int, Any] = {
            h: XGBClassifier(**params) for h in [6, 12, 24]
        }
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_infection: np.ndarray,
        y_deterioration: np.ndarray,
    ) -> None:
        """Train all horizon-specific classifiers.

        Args:
            X:                Feature matrix, shape (n_samples, n_features).
            y_infection:      Binary labels, shape (n_samples, 4) for horizons
                              [6h, 12h, 24h, 48h].
            y_deterioration:  Binary labels, shape (n_samples, 3) for horizons
                              [6h, 12h, 24h].
        """
        for i, h in enumerate([6, 12, 24, 48]):
            self.infection_models[h].fit(X, y_infection[:, i])

        for i, h in enumerate([6, 12, 24]):
            self.deterioration_models[h].fit(X, y_deterioration[:, i])

        self._is_fitted = True

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Predict risk probabilities for all horizons.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Dict with:
                'infection':    (n_samples, 4) — probabilities at 6/12/24/48h
                'deterioration':(n_samples, 3) — probabilities at 6/12/24h
        """
        if not self._is_fitted:
            raise RuntimeError("XGBoostBaseline must be fitted before predict_proba.")

        inf_preds = np.stack(
            [self.infection_models[h].predict_proba(X)[:, 1] for h in [6, 12, 24, 48]],
            axis=1,
        )
        det_preds = np.stack(
            [self.deterioration_models[h].predict_proba(X)[:, 1] for h in [6, 12, 24]],
            axis=1,
        )
        return {"infection": inf_preds, "deterioration": det_preds}
