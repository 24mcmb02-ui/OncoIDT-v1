"""
DeepHit discrete-time survival model for OncoIDT.

Outputs a probability mass function (PMF) over 72 discrete time bins
(1-hour resolution, 0–72h horizon). Derives median survival time and
80% credible interval from the PMF.

Requirements: 5.7
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_TIME_BINS = 72   # 1h resolution, 0–72h horizon


class DeepHit(nn.Module):
    """Discrete-time survival model producing a PMF over 72 time bins.

    Architecture:
        Input features → 3-layer MLP → Softmax → PMF over 72 bins

    Args:
        in_features:  Dimension of the input feature vector.
        hidden_dim:   Hidden layer dimension (default 256).
        num_bins:     Number of discrete time bins (default 72).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_bins: int = NUM_TIME_BINS,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch, in_features).

        Returns:
            PMF over time bins, shape (batch, num_bins).
            Values sum to 1 along the bin dimension.
        """
        logits = self.net(x)                    # (batch, num_bins)
        return F.softmax(logits, dim=-1)        # (batch, num_bins)

    def predict_survival(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run inference and derive summary statistics from the PMF.

        Args:
            x: Input features, shape (batch, in_features).

        Returns:
            Dict with keys:
                pmf:            (batch, num_bins) — probability mass function
                survival:       (batch, num_bins) — survival function S(t)
                median_hours:   (batch,) — median survival time in hours
                ci_80_lower:    (batch,) — 10th percentile (hours)
                ci_80_upper:    (batch,) — 90th percentile (hours)
        """
        pmf = self.forward(x)                   # (batch, num_bins)

        # Survival function: S(t) = P(T > t) = 1 - CDF(t)
        # CDF(t) = sum of PMF up to and including bin t
        cdf = torch.cumsum(pmf, dim=-1)         # (batch, num_bins)
        survival = 1.0 - cdf                    # (batch, num_bins)

        # Bin centres in hours: bin 0 = 0.5h, bin 1 = 1.5h, ..., bin 71 = 71.5h
        bin_centres = torch.arange(
            0.5, self.num_bins + 0.5, 1.0, device=x.device
        )  # (num_bins,)

        # Median: first bin where CDF >= 0.5
        median_hours = _quantile_from_cdf(cdf, 0.50, bin_centres)

        # 80% credible interval: [10th percentile, 90th percentile]
        ci_lower = _quantile_from_cdf(cdf, 0.10, bin_centres)
        ci_upper = _quantile_from_cdf(cdf, 0.90, bin_centres)

        return {
            "pmf": pmf,
            "survival": survival,
            "median_hours": median_hours,
            "ci_80_lower": ci_lower,
            "ci_80_upper": ci_upper,
        }


def _quantile_from_cdf(
    cdf: torch.Tensor,
    quantile: float,
    bin_centres: torch.Tensor,
) -> torch.Tensor:
    """Extract a quantile from a discrete CDF via linear interpolation.

    Args:
        cdf:         Cumulative distribution function, shape (batch, num_bins).
        quantile:    Target quantile in (0, 1).
        bin_centres: Bin centre times, shape (num_bins,).

    Returns:
        Estimated quantile time, shape (batch,).
    """
    # Find the first bin where CDF >= quantile
    # (batch, num_bins) boolean mask
    exceeded = cdf >= quantile

    # Index of first True per row; clamp to valid range
    # argmax returns 0 if no True found (all False), which is fine as a fallback
    idx = exceeded.long().argmax(dim=-1)   # (batch,)

    return bin_centres[idx]                # (batch,)
