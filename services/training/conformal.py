"""
Conformal prediction wrapper for OncoIDT risk score uncertainty estimation.

Provides distribution-free coverage guarantees without assumptions on the
underlying model architecture. Uses the split conformal prediction approach:

1. Calibrate on a held-out calibration set to compute the nonconformity
   quantile q̂_α at coverage level 1-α.
2. At inference time, produce prediction intervals:
   [ŷ - q̂_α, ŷ + q̂_α]

The marginal coverage guarantee is:
   P(y ∈ Ĉ(x)) ≥ 1 - α

for any (x, y) drawn from the same distribution as the calibration set,
provided |calibration set| ≥ 1/α.

Requirements: 6.5
"""
from __future__ import annotations

import math

import numpy as np


class ConformalPredictor:
    """Split conformal prediction wrapper for regression/probability outputs.

    Nonconformity score: s_i = |y_i - ŷ_i|  (absolute residual).

    Args:
        alpha: Miscoverage level. The predictor targets 1-alpha coverage.
               Default 0.1 → 90% coverage.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._quantile: float | None = None
        self._n_cal: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._quantile is not None

    def calibrate(self, cal_scores: np.ndarray) -> None:
        """Compute the nonconformity quantile from calibration set scores.

        Args:
            cal_scores: Nonconformity scores for the calibration set,
                        shape (n_cal,). Typically |y_i - ŷ_i| for each
                        calibration example.

        The quantile is computed as the ⌈(1-α)(1 + 1/n)⌉-th order statistic,
        which gives the finite-sample coverage guarantee.
        """
        cal_scores = np.asarray(cal_scores, dtype=float)
        if cal_scores.ndim != 1:
            raise ValueError("cal_scores must be a 1-D array")

        n = len(cal_scores)
        if n == 0:
            raise ValueError("Calibration set must not be empty")

        self._n_cal = n

        # Finite-sample corrected quantile level: ceil((1-alpha)*(n+1)) / n
        # This gives the smallest quantile level that guarantees 1-alpha coverage.
        level = math.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)   # clamp to [0, 1]

        self._quantile = float(np.quantile(cal_scores, level))

    def predict_interval(
        self,
        point_pred: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Produce a prediction interval around a point prediction.

        Args:
            point_pred: Point prediction(s), scalar or array of shape (n,).

        Returns:
            (lower, upper) tuple with the same shape as point_pred.
            lower = point_pred - q̂_α
            upper = point_pred + q̂_α

        Raises:
            RuntimeError: If ``calibrate`` has not been called yet.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "ConformalPredictor must be calibrated before calling "
                "predict_interval. Call calibrate(cal_scores) first."
            )

        q = self._quantile
        lower = np.asarray(point_pred) - q
        upper = np.asarray(point_pred) + q

        # Clamp to [0, 1] for probability outputs
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)

        # Return scalar if input was scalar
        if np.ndim(point_pred) == 0:
            return float(lower), float(upper)
        return lower, upper

    @property
    def quantile(self) -> float:
        """The calibrated nonconformity quantile q̂_α."""
        if self._quantile is None:
            raise RuntimeError("Not yet calibrated.")
        return self._quantile

    def __repr__(self) -> str:
        status = (
            f"calibrated (n={self._n_cal}, q={self._quantile:.4f})"
            if self.is_calibrated
            else "not calibrated"
        )
        return f"ConformalPredictor(alpha={self.alpha}, {status})"
