"""
OncoIDT Explainability Service — SHAP Computation Engine.

Provides:
  - XGBoostExplainer: SHAP TreeExplainer for XGBoost baseline model
  - NeuralExplainer:  SHAP DeepExplainer for Neural CDE; falls back to
                      KernelExplainer if DeepExplainer is unsupported
  - compute_top_features(): rank features by absolute SHAP value
  - compute_delta_explanation(): SHAP(counterfactual) - SHAP(baseline)

Requirements: 11.1, 11.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Ordered feature names matching the inference pipeline's feature vector layout
# (must stay in sync with services/inference/scorer.py ORDERED_KEYS)
FEATURE_NAMES: list[str] = [
    "anc",
    "wbc",
    "lymphocytes",
    "crp_mg_l",
    "procalcitonin_ug_l",
    "temperature_c",
    "heart_rate_bpm",
    "respiratory_rate_rpm",
    "sbp_mmhg",
    "dbp_mmhg",
    "spo2_pct",
    "gcs",
    "days_since_last_chemo_dose",
    "immunosuppression_score",
    "chemo_cycle_number",
    "vitals_mean_1h",
    "vitals_std_1h",
    "vitals_min_1h",
    "vitals_max_1h",
    "vitals_mean_6h",
    "vitals_std_6h",
    "vitals_min_6h",
    "vitals_max_6h",
    "vitals_mean_24h",
    "vitals_std_24h",
    "vitals_min_24h",
    "vitals_max_24h",
    "anc_slope_6h",
    "anc_slope_24h",
    "time_since_last_antibiotic_hours",
    "prior_infection_count",
    "antibiotic_active",
    "co_located_active_infections",
    "staff_contact_count_24h",
]

TOP_N = 5  # number of top features to return


@dataclass
class FeatureAttribution:
    """A single feature's SHAP attribution."""
    feature_name: str
    shap_value: float          # signed SHAP value
    abs_shap_value: float      # |shap_value|
    feature_value: float | None = None
    rank: int = 0              # 1 = most important


@dataclass
class ExplanationResult:
    """Full SHAP explanation for one inference output."""
    patient_id: str
    score_type: str            # "infection" | "deterioration"
    forecast_horizon_hours: int
    model_version: str
    top_features: list[FeatureAttribution] = field(default_factory=list)
    all_shap_values: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    is_rule_driven: bool = False
    rule_ids: list[str] = field(default_factory=list)


@dataclass
class DeltaExplanation:
    """Delta SHAP explanation for a counterfactual simulation."""
    patient_id: str
    score_type: str
    baseline_score: float
    counterfactual_score: float
    score_delta: float
    top_delta_features: list[FeatureAttribution] = field(default_factory=list)


def _features_dict_to_array(features: dict[str, Any]) -> np.ndarray:
    """Convert a feature dict to a numpy array aligned with FEATURE_NAMES."""
    return np.array(
        [float(features.get(k, 0.0) or 0.0) for k in FEATURE_NAMES],
        dtype=np.float32,
    )


def _rank_top_features(
    shap_values: np.ndarray,
    feature_values: np.ndarray | None,
    n: int = TOP_N,
) -> list[FeatureAttribution]:
    """Return the top-N features sorted by absolute SHAP value."""
    abs_vals = np.abs(shap_values)
    top_indices = np.argsort(abs_vals)[::-1][:n]

    result: list[FeatureAttribution] = []
    for rank, idx in enumerate(top_indices, start=1):
        fv = float(feature_values[idx]) if feature_values is not None else None
        result.append(FeatureAttribution(
            feature_name=FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}",
            shap_value=float(shap_values[idx]),
            abs_shap_value=float(abs_vals[idx]),
            feature_value=fv,
            rank=rank,
        ))
    return result


# ---------------------------------------------------------------------------
# XGBoost Explainer
# ---------------------------------------------------------------------------

class XGBoostExplainer:
    """
    SHAP TreeExplainer for the XGBoost baseline model.

    Usage:
        explainer = XGBoostExplainer(xgb_model)
        result = explainer.explain(features_dict, patient_id, "infection", 24, "v1.0")
    """

    def __init__(self, model: Any) -> None:
        """
        Args:
            model: A fitted XGBClassifier (or any tree model supported by
                   shap.TreeExplainer).
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for XGBoostExplainer. Install with: pip install shap"
            ) from exc

        import shap
        self._explainer = shap.TreeExplainer(model)
        self._model = model

    def explain(
        self,
        features: dict[str, Any],
        patient_id: str,
        score_type: str,
        forecast_horizon_hours: int,
        model_version: str,
        rule_ids: list[str] | None = None,
    ) -> ExplanationResult:
        """
        Compute SHAP values for a single patient feature vector.

        Args:
            features:              Feature dict (keys matching FEATURE_NAMES).
            patient_id:            Patient identifier.
            score_type:            "infection" or "deterioration".
            forecast_horizon_hours: Forecast horizon in hours.
            model_version:         Model version string for provenance.
            rule_ids:              List of rule IDs that fired (if any).

        Returns:
            ExplanationResult with top-5 features by absolute SHAP value.
        """
        x = _features_dict_to_array(features).reshape(1, -1)
        shap_values = self._explainer.shap_values(x)

        # TreeExplainer may return a list (one array per class) for classifiers
        if isinstance(shap_values, list):
            # Use the positive-class SHAP values (index 1)
            sv = np.array(shap_values[1]).flatten()
        else:
            sv = np.array(shap_values).flatten()

        top_features = _rank_top_features(sv, x.flatten())

        return ExplanationResult(
            patient_id=patient_id,
            score_type=score_type,
            forecast_horizon_hours=forecast_horizon_hours,
            model_version=model_version,
            top_features=top_features,
            all_shap_values=sv.tolist(),
            feature_names=FEATURE_NAMES,
            is_rule_driven=bool(rule_ids),
            rule_ids=rule_ids or [],
        )

    def explain_delta(
        self,
        baseline_features: dict[str, Any],
        counterfactual_features: dict[str, Any],
        patient_id: str,
        score_type: str,
        baseline_score: float,
        counterfactual_score: float,
    ) -> DeltaExplanation:
        """
        Compute delta SHAP: SHAP(counterfactual) - SHAP(baseline) per feature.

        Requirements: 11.5
        """
        x_base = _features_dict_to_array(baseline_features).reshape(1, -1)
        x_cf = _features_dict_to_array(counterfactual_features).reshape(1, -1)

        sv_base_raw = self._explainer.shap_values(x_base)
        sv_cf_raw = self._explainer.shap_values(x_cf)

        sv_base = np.array(sv_base_raw[1] if isinstance(sv_base_raw, list) else sv_base_raw).flatten()
        sv_cf = np.array(sv_cf_raw[1] if isinstance(sv_cf_raw, list) else sv_cf_raw).flatten()

        delta_sv = sv_cf - sv_base
        top_delta = _rank_top_features(delta_sv, x_cf.flatten() - x_base.flatten())

        return DeltaExplanation(
            patient_id=patient_id,
            score_type=score_type,
            baseline_score=baseline_score,
            counterfactual_score=counterfactual_score,
            score_delta=counterfactual_score - baseline_score,
            top_delta_features=top_delta,
        )


# ---------------------------------------------------------------------------
# Neural Explainer (DeepExplainer with KernelExplainer fallback)
# ---------------------------------------------------------------------------

class NeuralExplainer:
    """
    SHAP explainer for Neural CDE and other PyTorch models.

    Attempts to use DeepExplainer; falls back to KernelExplainer if the
    model architecture is not supported (e.g., custom ODE layers).

    Usage:
        explainer = NeuralExplainer(model, background_data)
        result = explainer.explain(features_dict, patient_id, "infection", 24, "v1.0")
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        use_kernel_fallback: bool = False,
    ) -> None:
        """
        Args:
            model:              PyTorch model with a predict_numpy() method or
                                a callable that accepts a numpy array and returns
                                a numpy array of predictions.
            background_data:    Background dataset for SHAP, shape
                                (n_background, n_features). Typically a random
                                sample of 50–200 training examples.
            use_kernel_fallback: Force KernelExplainer even if DeepExplainer
                                 would succeed (useful for testing).
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for NeuralExplainer. Install with: pip install shap"
            ) from exc

        import shap

        self._model = model
        self._background = background_data
        self._explainer: Any = None
        self._use_kernel = use_kernel_fallback

        if not use_kernel_fallback:
            self._explainer = self._try_deep_explainer(shap, model, background_data)

        if self._explainer is None:
            logger.info(
                "NeuralExplainer: DeepExplainer not supported for this model; "
                "falling back to KernelExplainer."
            )
            self._explainer = self._build_kernel_explainer(shap, model, background_data)
            self._use_kernel = True

    @staticmethod
    def _try_deep_explainer(
        shap: Any,
        model: Any,
        background: np.ndarray,
    ) -> Any | None:
        """Attempt to build a DeepExplainer; return None on failure."""
        try:
            import torch

            # Wrap numpy background as a torch tensor
            bg_tensor = torch.tensor(background, dtype=torch.float32)

            # Build a wrapper that returns the first output (infection risk at 6h)
            # for SHAP attribution purposes
            class _ModelWrapper(torch.nn.Module):
                def __init__(self, inner: Any) -> None:
                    super().__init__()
                    self._inner = inner

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out = self._inner(x)
                    # Handle tuple outputs (infection_risk, deterioration_risk, ...)
                    if isinstance(out, (tuple, list)):
                        return out[0][:, 0:1]  # first horizon of infection risk
                    return out

            wrapped = _ModelWrapper(model)
            explainer = shap.DeepExplainer(wrapped, bg_tensor)
            return explainer
        except Exception as exc:
            logger.debug("DeepExplainer construction failed: %s", exc)
            return None

    @staticmethod
    def _build_kernel_explainer(
        shap: Any,
        model: Any,
        background: np.ndarray,
    ) -> Any:
        """Build a KernelExplainer using a numpy-compatible predict function."""

        def _predict_fn(x: np.ndarray) -> np.ndarray:
            """Wrapper that calls model.predict_numpy or falls back to __call__."""
            if hasattr(model, "predict_numpy"):
                return model.predict_numpy(x)
            try:
                import torch
                with torch.no_grad():
                    t = torch.tensor(x, dtype=torch.float32)
                    out = model(t)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    return out.numpy()[:, 0]
            except Exception:
                # Last resort: return zeros
                return np.zeros(len(x), dtype=np.float32)

        # Use k-means summary of background to keep KernelExplainer tractable
        background_summary = shap.kmeans(background, min(50, len(background)))
        return shap.KernelExplainer(_predict_fn, background_summary)

    def explain(
        self,
        features: dict[str, Any],
        patient_id: str,
        score_type: str,
        forecast_horizon_hours: int,
        model_version: str,
        rule_ids: list[str] | None = None,
    ) -> ExplanationResult:
        """
        Compute SHAP values for a single patient feature vector.

        Returns:
            ExplanationResult with top-5 features by absolute SHAP value.
        """
        import shap as _shap

        x = _features_dict_to_array(features).reshape(1, -1)

        try:
            if self._use_kernel:
                sv_raw = self._explainer.shap_values(x, nsamples=100, silent=True)
            else:
                import torch
                x_tensor = torch.tensor(x, dtype=torch.float32)
                sv_raw = self._explainer.shap_values(x_tensor)
        except Exception as exc:
            logger.warning("SHAP computation failed for patient=%s: %s", patient_id, exc)
            # Return zero attributions rather than crashing
            sv_raw = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        # Normalise to 1-D array
        if isinstance(sv_raw, list):
            sv = np.array(sv_raw[0]).flatten()
        else:
            sv = np.array(sv_raw).flatten()

        # Pad or truncate to match FEATURE_NAMES length
        n = len(FEATURE_NAMES)
        if len(sv) < n:
            sv = np.pad(sv, (0, n - len(sv)))
        else:
            sv = sv[:n]

        top_features = _rank_top_features(sv, x.flatten())

        return ExplanationResult(
            patient_id=patient_id,
            score_type=score_type,
            forecast_horizon_hours=forecast_horizon_hours,
            model_version=model_version,
            top_features=top_features,
            all_shap_values=sv.tolist(),
            feature_names=FEATURE_NAMES,
            is_rule_driven=bool(rule_ids),
            rule_ids=rule_ids or [],
        )

    def explain_delta(
        self,
        baseline_features: dict[str, Any],
        counterfactual_features: dict[str, Any],
        patient_id: str,
        score_type: str,
        baseline_score: float,
        counterfactual_score: float,
    ) -> DeltaExplanation:
        """
        Compute delta SHAP: SHAP(counterfactual) - SHAP(baseline) per feature.

        Requirements: 11.5
        """
        import shap as _shap

        x_base = _features_dict_to_array(baseline_features).reshape(1, -1)
        x_cf = _features_dict_to_array(counterfactual_features).reshape(1, -1)

        def _get_sv(x: np.ndarray) -> np.ndarray:
            try:
                if self._use_kernel:
                    sv_raw = self._explainer.shap_values(x, nsamples=100, silent=True)
                else:
                    import torch
                    x_t = torch.tensor(x, dtype=torch.float32)
                    sv_raw = self._explainer.shap_values(x_t)
            except Exception as exc:
                logger.warning("SHAP delta computation failed: %s", exc)
                return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

            sv = np.array(sv_raw[0] if isinstance(sv_raw, list) else sv_raw).flatten()
            n = len(FEATURE_NAMES)
            return sv[:n] if len(sv) >= n else np.pad(sv, (0, n - len(sv)))

        sv_base = _get_sv(x_base)
        sv_cf = _get_sv(x_cf)
        delta_sv = sv_cf - sv_base

        top_delta = _rank_top_features(delta_sv, x_cf.flatten() - x_base.flatten())

        return DeltaExplanation(
            patient_id=patient_id,
            score_type=score_type,
            baseline_score=baseline_score,
            counterfactual_score=counterfactual_score,
            score_delta=counterfactual_score - baseline_score,
            top_delta_features=top_delta,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_explainer(
    model: Any,
    model_type: str,
    background_data: np.ndarray | None = None,
) -> XGBoostExplainer | NeuralExplainer:
    """
    Factory that returns the appropriate explainer for a given model type.

    Args:
        model:          The trained model object.
        model_type:     "xgboost" | "neural_cde" | "lstm"
        background_data: Required for neural explainers.

    Returns:
        XGBoostExplainer or NeuralExplainer instance.
    """
    if model_type == "xgboost":
        return XGBoostExplainer(model)

    if background_data is None:
        raise ValueError(
            "background_data is required for neural explainers. "
            "Provide a sample of training feature vectors."
        )
    return NeuralExplainer(model, background_data)
