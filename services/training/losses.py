"""
Multi-task loss functions for OncoIDT training.

- FocalLoss(gamma=2, alpha=0.75) for infection and deterioration heads
- WeibullNLLLoss for the survival head
- MultiTaskLoss combining all three with configurable task weights

Requirements: 6.3
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2). Higher values down-weight
               easy examples more aggressively.
        alpha: Class weight for the positive class (default 0.75).
               The negative class receives weight (1 - alpha).
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw model outputs (before sigmoid), shape (...).
                     Also accepts probabilities — values in (0, 1) are treated
                     as probabilities; values outside are treated as logits.
            targets: Binary labels in {0, 1}, same shape as logits.

        Returns:
            Scalar loss (if reduction='mean' or 'sum').
        """
        # BCE with logits for numerical stability
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability of the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t: per-sample class weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeibullNLLLoss(nn.Module):
    """Negative log-likelihood of the Weibull distribution.

    The Weibull PDF is:
        f(t | k, λ) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)

    NLL = -log f(t | k, λ)
        = log(λ) - log(k) + (1-k)*log(t/λ) + (t/λ)^k

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        params: torch.Tensor,
        event_times: torch.Tensor,
        event_observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            params:         Weibull parameters, shape (batch, 2).
                            params[:, 0] = shape k  (> 0)
                            params[:, 1] = scale λ  (> 0)
            event_times:    Observed or censored times, shape (batch,).
            event_observed: Binary mask; 1 = event observed, 0 = censored.
                            If None, all events are treated as observed.

        Returns:
            Mean NLL loss.
        """
        k = params[:, 0].clamp(min=self.eps)
        lam = params[:, 1].clamp(min=self.eps)
        t = event_times.clamp(min=self.eps)

        # Log-likelihood of Weibull
        log_pdf = (
            torch.log(k)
            - torch.log(lam)
            + (k - 1) * torch.log(t / lam)
            - (t / lam) ** k
        )

        if event_observed is not None:
            # For censored observations use the survival function: S(t) = exp(-(t/λ)^k)
            log_sf = -((t / lam) ** k)
            log_likelihood = event_observed * log_pdf + (1 - event_observed) * log_sf
        else:
            log_likelihood = log_pdf

        return -log_likelihood.mean()


class MultiTaskLoss(nn.Module):
    """Weighted combination of infection, deterioration, and survival losses.

    L_total = λ_infection * L_infection
            + λ_deterioration * L_deterioration
            + λ_survival * L_survival

    Args:
        lambda_infection:    Weight for infection focal loss (default 1.0).
        lambda_deterioration: Weight for deterioration focal loss (default 0.8).
        lambda_survival:     Weight for survival NLL loss (default 0.3).
        focal_gamma:         Focal loss gamma parameter (default 2).
        focal_alpha:         Focal loss alpha parameter (default 0.75).
    """

    def __init__(
        self,
        lambda_infection: float = 1.0,
        lambda_deterioration: float = 0.8,
        lambda_survival: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ) -> None:
        super().__init__()
        self.lambda_infection = lambda_infection
        self.lambda_deterioration = lambda_deterioration
        self.lambda_survival = lambda_survival

        self.infection_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.deterioration_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.survival_loss = WeibullNLLLoss()

    def forward(
        self,
        infection_logits: torch.Tensor,
        infection_targets: torch.Tensor,
        deterioration_logits: torch.Tensor,
        deterioration_targets: torch.Tensor,
        survival_params: torch.Tensor,
        event_times: torch.Tensor,
        event_observed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            infection_logits:       (batch, 4) — raw logits for 4 horizons
            infection_targets:      (batch, 4) — binary labels
            deterioration_logits:   (batch, 3) — raw logits for 3 horizons
            deterioration_targets:  (batch, 3) — binary labels
            survival_params:        (batch, 2) — Weibull (shape, scale)
            event_times:            (batch,)   — time to event or censoring
            event_observed:         (batch,)   — 1 if event observed, 0 if censored

        Returns:
            total_loss: Scalar weighted sum.
            components: Dict with individual loss values for logging.
        """
        l_inf = self.infection_loss(infection_logits, infection_targets)
        l_det = self.deterioration_loss(deterioration_logits, deterioration_targets)
        l_surv = self.survival_loss(survival_params, event_times, event_observed)

        total = (
            self.lambda_infection * l_inf
            + self.lambda_deterioration * l_det
            + self.lambda_survival * l_surv
        )

        components = {
            "loss_infection": l_inf,
            "loss_deterioration": l_det,
            "loss_survival": l_surv,
            "loss_total": total,
        }
        return total, components
