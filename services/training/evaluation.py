"""
OncoIDT Training Pipeline — Evaluation Metrics.

Comprehensive evaluation suite for clinical risk prediction models:
- AUROC, AUPRC, Brier score, ECE, sensitivity/specificity at fixed operating points
- NRI / IDI vs NEWS2 baseline
- DeLong test for AUROC comparison
- Decision curve analysis (net benefit)
- Subgroup stratified metrics

Requirements: 16.1, 16.3, 16.4, 16.5, 16.7
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute the full evaluation metric suite for a binary classification task.

    Args:
        y_true: Ground truth binary labels, shape (n,).
        y_pred: Binary predictions (thresholded), shape (n,).
        y_prob: Predicted probabilities for the positive class, shape (n,).
        n_bins: Number of bins for ECE computation (default 10).

    Returns:
        Dict with keys:
            auroc, auprc, brier_score, ece,
            sensitivity_at_90spec, specificity_at_80sens,
            precision, recall, f1, accuracy
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob).ravel()

    metrics: dict[str, float] = {}

    # AUROC
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")

    # AUPRC
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = float("nan")

    # Brier score
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # ECE (Expected Calibration Error)
    metrics["ece"] = _compute_ece(y_true, y_prob, n_bins=n_bins)

    # Sensitivity at 90% specificity
    metrics["sensitivity_at_90spec"] = _sensitivity_at_specificity(y_true, y_prob, target_spec=0.90)

    # Specificity at 80% sensitivity
    metrics["specificity_at_80sens"] = _specificity_at_sensitivity(y_true, y_prob, target_sens=0.80)

    # Standard classification metrics at default threshold
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    return metrics


def _compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error with uniform binning.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width bins.

    Returns:
        ECE scalar in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

        if mask.sum() == 0:
            continue

        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def _sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_spec: float = 0.90,
) -> float:
    """Return sensitivity at the threshold that achieves target specificity."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr

    # Find threshold where specificity >= target
    valid = specificity >= target_spec
    if not valid.any():
        return float("nan")

    # Among valid thresholds, pick the one with highest sensitivity
    best_tpr = float(tpr[valid].max())
    return best_tpr


def _specificity_at_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sens: float = 0.80,
) -> float:
    """Return specificity at the threshold that achieves target sensitivity."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr

    # Find threshold where sensitivity >= target
    valid = tpr >= target_sens
    if not valid.any():
        return float("nan")

    # Among valid thresholds, pick the one with highest specificity
    best_spec = float(specificity[valid].max())
    return best_spec


# ---------------------------------------------------------------------------
# NRI / IDI
# ---------------------------------------------------------------------------

def compute_nri_idi(
    y_true: np.ndarray,
    y_prob_new: np.ndarray,
    y_prob_ref: np.ndarray,
) -> dict[str, float]:
    """
    Compute Net Reclassification Improvement (NRI) and Integrated
    Discrimination Improvement (IDI) of a new model vs a reference model.

    NRI measures the net proportion of events correctly reclassified upward
    and non-events correctly reclassified downward.

    IDI = (mean(new_prob | event) - mean(new_prob | non-event))
        - (mean(ref_prob | event) - mean(ref_prob | non-event))

    Args:
        y_true:     Binary ground truth labels, shape (n,).
        y_prob_new: Predicted probabilities from the new model, shape (n,).
        y_prob_ref: Predicted probabilities from the reference model, shape (n,).

    Returns:
        Dict with keys: nri, nri_events, nri_non_events, idi,
                        idi_new_is, idi_ref_is
    """
    y_true = np.asarray(y_true).ravel()
    y_prob_new = np.asarray(y_prob_new).ravel()
    y_prob_ref = np.asarray(y_prob_ref).ravel()

    events = y_true == 1
    non_events = y_true == 0

    # NRI (continuous, no category boundaries)
    delta = y_prob_new - y_prob_ref

    # Events: positive reclassification = new > ref
    nri_events = float(np.mean(delta[events] > 0) - np.mean(delta[events] < 0))

    # Non-events: positive reclassification = new < ref
    nri_non_events = float(np.mean(delta[non_events] < 0) - np.mean(delta[non_events] > 0))

    nri = nri_events + nri_non_events

    # IDI
    idi_new_is = float(np.mean(y_prob_new[events]) - np.mean(y_prob_new[non_events]))
    idi_ref_is = float(np.mean(y_prob_ref[events]) - np.mean(y_prob_ref[non_events]))
    idi = idi_new_is - idi_ref_is

    return {
        "nri": nri,
        "nri_events": nri_events,
        "nri_non_events": nri_non_events,
        "idi": idi,
        "idi_new_is": idi_new_is,
        "idi_ref_is": idi_ref_is,
    }


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------

def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> dict[str, float]:
    """
    DeLong test for comparing two AUROC values.

    Implements the DeLong et al. (1988) method for computing the variance
    of the difference between two correlated AUROCs.

    Args:
        y_true:   Binary ground truth labels, shape (n,).
        y_prob_a: Predicted probabilities from model A, shape (n,).
        y_prob_b: Predicted probabilities from model B, shape (n,).

    Returns:
        Dict with keys: auroc_a, auroc_b, auroc_diff, z_stat, p_value
    """
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel()
    y_prob_a = np.asarray(y_prob_a).ravel()
    y_prob_b = np.asarray(y_prob_b).ravel()

    auroc_a = float(roc_auc_score(y_true, y_prob_a))
    auroc_b = float(roc_auc_score(y_true, y_prob_b))

    # Compute structural components for DeLong variance
    var_a, var_b, cov_ab = _delong_variance(y_true, y_prob_a, y_prob_b)

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return {
            "auroc_a": auroc_a,
            "auroc_b": auroc_b,
            "auroc_diff": auroc_a - auroc_b,
            "z_stat": float("nan"),
            "p_value": float("nan"),
        }

    z_stat = (auroc_a - auroc_b) / np.sqrt(var_diff)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return {
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "auroc_diff": auroc_a - auroc_b,
        "z_stat": float(z_stat),
        "p_value": p_value,
    }


def _delong_variance(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute variance and covariance of two AUROCs using the DeLong method.

    Returns: (var_a, var_b, cov_ab)
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0, 0.0

    # Placement values: V_10 and V_01 for each model
    def placement_values(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute V_10 (for positives) and V_01 (for negatives)."""
        v10 = np.zeros(n_pos)
        v01 = np.zeros(n_neg)
        for i, pi in enumerate(pos_idx):
            v10[i] = np.mean(scores[pi] > scores[neg_idx]) + 0.5 * np.mean(scores[pi] == scores[neg_idx])
        for j, ni in enumerate(neg_idx):
            v01[j] = np.mean(scores[ni] < scores[pos_idx]) + 0.5 * np.mean(scores[ni] == scores[pos_idx])
        return v10, v01

    v10_a, v01_a = placement_values(y_prob_a)
    v10_b, v01_b = placement_values(y_prob_b)

    # Structural components
    s10_aa = float(np.var(v10_a, ddof=1)) if n_pos > 1 else 0.0
    s01_aa = float(np.var(v01_a, ddof=1)) if n_neg > 1 else 0.0
    s10_bb = float(np.var(v10_b, ddof=1)) if n_pos > 1 else 0.0
    s01_bb = float(np.var(v01_b, ddof=1)) if n_neg > 1 else 0.0
    s10_ab = float(np.cov(v10_a, v10_b)[0, 1]) if n_pos > 1 else 0.0
    s01_ab = float(np.cov(v01_a, v01_b)[0, 1]) if n_neg > 1 else 0.0

    var_a = s10_aa / n_pos + s01_aa / n_neg
    var_b = s10_bb / n_pos + s01_bb / n_neg
    cov_ab = s10_ab / n_pos + s01_ab / n_neg

    return var_a, var_b, cov_ab


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """
    Decision curve analysis: compute net benefit across probability thresholds.

    Net benefit = (TP/n) - (FP/n) * (pt / (1 - pt))

    where pt is the probability threshold (decision threshold).

    Args:
        y_true:     Binary ground truth labels, shape (n,).
        y_prob:     Predicted probabilities, shape (n,).
        thresholds: Array of probability thresholds to evaluate.
                    Defaults to np.linspace(0.01, 0.99, 99).

    Returns:
        List of dicts with keys: threshold, net_benefit, net_benefit_all,
        net_benefit_none, tp_rate, fp_rate
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    prevalence = float(y_true.mean())

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    results: list[dict[str, float]] = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())

        net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))

        # Treat-all strategy
        net_benefit_all = prevalence - (1 - prevalence) * (pt / (1 - pt))

        # Treat-none strategy: always 0
        net_benefit_none = 0.0

        results.append({
            "threshold": float(pt),
            "net_benefit": float(net_benefit),
            "net_benefit_all": float(net_benefit_all),
            "net_benefit_none": net_benefit_none,
            "tp_rate": float(tp / max(1, y_true.sum())),
            "fp_rate": float(fp / max(1, (1 - y_true).sum())),
        })

    return results


# ---------------------------------------------------------------------------
# Subgroup metrics
# ---------------------------------------------------------------------------

def subgroup_metrics(
    df: pd.DataFrame,
    stratify_by: str,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
    min_group_size: int = 10,
) -> list[dict[str, Any]]:
    """
    Compute evaluation metrics per subgroup.

    Args:
        df:             DataFrame with columns for labels, predictions, and
                        the stratification variable.
        stratify_by:    Column name to stratify by (e.g. "chemo_regimen",
                        "anc_nadir_group", "age_group", "ward_zone").
        y_true_col:     Column name for ground truth labels.
        y_prob_col:     Column name for predicted probabilities.
        y_pred_col:     Column name for binary predictions.
        min_group_size: Minimum group size to compute metrics (default 10).

    Returns:
        List of dicts, one per subgroup, with subgroup label and all metrics.
    """
    if stratify_by not in df.columns:
        logger.warning("Stratification column '%s' not found in DataFrame.", stratify_by)
        return []

    results: list[dict[str, Any]] = []

    for group_val, group_df in df.groupby(stratify_by):
        if len(group_df) < min_group_size:
            logger.debug(
                "Skipping subgroup %s=%s: only %d samples (min=%d)",
                stratify_by, group_val, len(group_df), min_group_size,
            )
            continue

        y_true = group_df[y_true_col].values
        y_prob = group_df[y_prob_col].values
        y_pred = group_df[y_pred_col].values

        try:
            group_metrics = compute_metrics(y_true, y_pred, y_prob)
        except Exception as exc:
            logger.warning(
                "Metrics failed for subgroup %s=%s: %s", stratify_by, group_val, exc
            )
            group_metrics = {}

        row: dict[str, Any] = {
            "subgroup_variable": stratify_by,
            "subgroup_value": str(group_val),
            "n_samples": len(group_df),
            "n_events": int(y_true.sum()),
            "event_rate": float(y_true.mean()),
        }
        row.update(group_metrics)
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Convenience: compute all metrics for a model run
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_prob_ref: np.ndarray | None = None,
    df_subgroups: pd.DataFrame | None = None,
    subgroup_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the full evaluation suite for a single model.

    Args:
        y_true:        Ground truth binary labels.
        y_prob:        Predicted probabilities.
        threshold:     Decision threshold for binary predictions.
        y_prob_ref:    Reference model probabilities for NRI/IDI and DeLong.
        df_subgroups:  DataFrame for subgroup analysis (must include y_true, y_prob).
        subgroup_cols: List of columns to stratify by.

    Returns:
        Dict with keys: metrics, nri_idi (if ref provided),
        delong (if ref provided), subgroups (if df provided),
        decision_curve
    """
    y_pred = (y_prob >= threshold).astype(int)
    result: dict[str, Any] = {
        "metrics": compute_metrics(y_true, y_pred, y_prob),
        "decision_curve": decision_curve_analysis(y_true, y_prob),
    }

    if y_prob_ref is not None:
        result["nri_idi"] = compute_nri_idi(y_true, y_prob, y_prob_ref)
        result["delong"] = delong_test(y_true, y_prob, y_prob_ref)

    if df_subgroups is not None and subgroup_cols:
        result["subgroups"] = {}
        for col in subgroup_cols:
            result["subgroups"][col] = subgroup_metrics(df_subgroups, stratify_by=col)

    return result
