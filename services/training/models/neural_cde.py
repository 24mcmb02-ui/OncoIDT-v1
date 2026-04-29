"""
Neural Controlled Differential Equation model for OncoIDT.

Models patient state as a continuous-time dynamical system driven by a
control path derived from the observed clinical time series, handling
irregular sampling natively without imputation.

  z(t) = z(t₀) + ∫[t₀,t] f_θ(z(s)) dX(s)

where X(t) is the natural cubic spline of observed clinical measurements
and f_θ is a 3-layer MLP (the vector field).

Requirements: 6.1, 6.3, 5.4
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import torchcde
    # Compatibility shim: newer torchcde renamed these
    if not hasattr(torchcde, 'NaturalCubicSpline') and hasattr(torchcde, 'CubicSpline'):
        torchcde.NaturalCubicSpline = torchcde.CubicSpline
    if not hasattr(torchcde, 'natural_cubic_coeffs') and hasattr(torchcde, 'natural_cubic_spline_coeffs'):
        torchcde.natural_cubic_coeffs = torchcde.natural_cubic_spline_coeffs
    _TORCHCDE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCHCDE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vector field f_θ
# ---------------------------------------------------------------------------

class CDEFunc(nn.Module):
    """3-layer MLP vector field for the Neural CDE.

    Maps hidden state z ∈ R^hidden_dim to a matrix in
    R^(hidden_dim × input_dim) that is contracted with dX/dt.

    Architecture: input_dim → 256 → 256 → input_dim × hidden_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, input_dim * hidden_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        # z: (batch, hidden_dim)
        out = self.net(z)                                   # (batch, input_dim * hidden_dim)
        return out.view(*z.shape[:-1], self.hidden_dim, self.input_dim)


# ---------------------------------------------------------------------------
# Task-specific output heads
# ---------------------------------------------------------------------------

class InfectionHead(nn.Module):
    """Predicts infection risk at 4 forecast horizons (6h, 12h, 24h, 48h)."""

    def __init__(self, repr_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(repr_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))   # (batch, 4)


class DeteriorationHead(nn.Module):
    """Predicts deterioration risk at 3 forecast horizons (6h, 12h, 24h)."""

    def __init__(self, repr_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(repr_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))   # (batch, 3)


class SurvivalHead(nn.Module):
    """Outputs Weibull (shape k, scale λ) parameters for survival estimation."""

    def __init__(self, repr_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(repr_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softplus ensures positivity for both shape and scale
        return nn.functional.softplus(self.fc(x))   # (batch, 2)


# ---------------------------------------------------------------------------
# Shared representation layer
# ---------------------------------------------------------------------------

class SharedRepresentation(nn.Module):
    """Linear(256, 256) + LayerNorm + ReLU shared across all task heads."""

    def __init__(self, in_dim: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Neural CDE model
# ---------------------------------------------------------------------------

class NeuralCDE(nn.Module):
    """Neural CDE patient state model with multi-task output heads.

    Args:
        input_dim:  Number of clinical features (channels in the time series).
        hidden_dim: Dimension of the CDE hidden state (default 256).

    Forward inputs:
        coeffs: Natural cubic spline coefficients produced by
                ``torchcde.natural_cubic_coeffs(x)``.
                Shape: (batch, seq_len, input_dim * 4) for cubic splines.

    Forward outputs:
        infection_risk:    (batch, 4)  — probabilities at 6h/12h/24h/48h
        deterioration_risk:(batch, 3)  — probabilities at 6h/12h/24h
        survival_params:   (batch, 2)  — Weibull (shape, scale)
        repr:              (batch, 256) — shared representation (for fusion)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if not _TORCHCDE_AVAILABLE:
            raise ImportError(
                "torchcde is required for NeuralCDE. "
                "Install with: pip install torchcde"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initial hidden state projection
        self.initial_linear = nn.Linear(input_dim, hidden_dim)

        # CDE vector field
        self.cde_func = CDEFunc(input_dim, hidden_dim)

        # Shared representation
        self.shared_repr = SharedRepresentation(hidden_dim, 256)

        # Task heads
        self.infection_head = InfectionHead(256)
        self.deterioration_head = DeteriorationHead(256)
        self.survival_head = SurvivalHead(256)

    def forward(
        self,
        coeffs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build the spline control path from pre-computed coefficients
        X = torchcde.NaturalCubicSpline(coeffs)

        # Initial hidden state from first observation
        z0 = self.initial_linear(X.evaluate(X.interval[0]))  # (batch, hidden_dim)

        # Integrate the CDE using Dormand-Prince solver with adjoint method
        z_T = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=X.interval,
            method="dopri5",
            adjoint=True,
            rtol=1e-3,
            atol=1e-5,
        )
        # z_T shape: (batch, 2, hidden_dim) — take the final time point
        z_final = z_T[:, -1]   # (batch, hidden_dim)

        # Shared representation
        repr_vec = self.shared_repr(z_final)   # (batch, 256)

        # Task heads
        infection_risk = self.infection_head(repr_vec)
        deterioration_risk = self.deterioration_head(repr_vec)
        survival_params = self.survival_head(repr_vec)

        return infection_risk, deterioration_risk, survival_params, repr_vec

    @staticmethod
    def prepare_coefficients(
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute natural cubic spline coefficients from a time series.

        Args:
            x: Observed clinical features, shape (batch, seq_len, input_dim).
               May contain NaN for missing observations.
            t: Optional observation times, shape (seq_len,).
               If None, assumes uniform spacing [0, 1, ..., seq_len-1].

        Returns:
            Spline coefficients suitable for ``NaturalCubicSpline``.
        """
        if t is not None:
            return torchcde.natural_cubic_coeffs(x, t)
        return torchcde.natural_cubic_coeffs(x)
