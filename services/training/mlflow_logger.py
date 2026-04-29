"""
OncoIDT Training Pipeline — MLflow Experiment Logging.

Centralised logging helper for all training runs. Handles:
- Hyperparameters and dataset metadata
- Per-fold and aggregate metrics
- Model artifact with SHA-256 checksum
- PNG artifacts: calibration plot, ROC curve
- CSV artifacts: ablation table, subgroup analysis, NRI/IDI, DeLong results
- Markdown evaluation report

Requirements: 14.3, 22.6
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow import guard
# ---------------------------------------------------------------------------

def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for experiment logging. "
            "Install with: pip install mlflow"
        ) from exc


# ---------------------------------------------------------------------------
# OncoMLflowLogger
# ---------------------------------------------------------------------------

class OncoMLflowLogger:
    """
    Wrapper around MLflow for OncoIDT training runs.

    Usage:
        logger = OncoMLflowLogger(experiment_name="oncoidt", tracking_uri="http://mlflow:5000")
        with logger.start_run(run_name="neural_cde_v1") as run_id:
            logger.log_params({"lr": 1e-3, "hidden_dim": 256})
            logger.log_metrics({"auroc": 0.85}, step=0)
            logger.log_model(model, "neural_cde")
    """

    def __init__(
        self,
        experiment_name: str = "oncoidt",
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        mlflow = _get_mlflow()
        self.mlflow = mlflow
        self.experiment_name = experiment_name
        self.tags = tags or {}

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        self._active_run_id: str | None = None

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> "OncoMLflowLogger":
        """Start an MLflow run. Returns self for use as context manager."""
        merged_tags = {**self.tags, **(tags or {})}
        run = self.mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested,
            tags=merged_tags,
        )
        self._active_run_id = run.info.run_id
        logger.info("MLflow run started: %s (id=%s)", run_name, self._active_run_id)
        return self

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run."""
        self.mlflow.end_run(status=status)
        logger.info("MLflow run ended: %s", self._active_run_id)
        self._active_run_id = None

    def __enter__(self) -> "OncoMLflowLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)

    @property
    def run_id(self) -> str | None:
        return self._active_run_id

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters. Flattens nested dicts with dot notation."""
        flat = _flatten_dict(params)
        # MLflow param values must be strings ≤ 500 chars
        str_params = {k: str(v)[:500] for k, v in flat.items()}
        self.mlflow.log_params(str_params)

    def log_dataset_metadata(
        self,
        metadata: dict[str, Any],
        feature_version: str,
        cohort_filter: dict[str, Any] | None = None,
    ) -> None:
        """Log dataset version and cohort filter as MLflow params."""
        params: dict[str, Any] = {
            "dataset.feature_version": feature_version,
            "dataset.n_samples": metadata.get("n_samples"),
            "dataset.n_features": metadata.get("n_features"),
            "dataset.n_patients": metadata.get("n_patients"),
            "dataset.event_rate": round(float(metadata.get("event_rate", 0)), 4),
            "dataset.train_size": metadata.get("train_size"),
            "dataset.val_size": metadata.get("val_size"),
            "dataset.test_size": metadata.get("test_size"),
            "dataset.label_type": metadata.get("label_type"),
        }
        if cohort_filter:
            params["dataset.cohort_start"] = str(cohort_filter.get("start_date", ""))
            params["dataset.cohort_end"] = str(cohort_filter.get("end_date", ""))
            params["dataset.ward_ids"] = str(cohort_filter.get("ward_ids", []))
        self.log_params(params)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None and not np.isnan(v)}
        self.mlflow.log_metrics(clean, step=step)

    def log_fold_metrics(
        self,
        fold_metrics: list[dict[str, float]],
        prefix: str = "cv",
    ) -> None:
        """
        Log per-fold metrics and aggregate (mean ± std) summary.

        Args:
            fold_metrics: List of metric dicts, one per fold.
            prefix:       Metric name prefix (e.g. "cv").
        """
        if not fold_metrics:
            return

        # Log each fold
        for i, fold_m in enumerate(fold_metrics):
            self.log_metrics(
                {f"{prefix}_fold{i}_{k}": v for k, v in fold_m.items()},
                step=i,
            )

        # Aggregate
        all_keys = set().union(*[m.keys() for m in fold_metrics])
        agg: dict[str, float] = {}
        for key in all_keys:
            vals = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
            if vals:
                agg[f"{prefix}_mean_{key}"] = float(np.mean(vals))
                agg[f"{prefix}_std_{key}"] = float(np.std(vals))
        self.log_metrics(agg)

    # ------------------------------------------------------------------
    # Model artifact
    # ------------------------------------------------------------------

    def log_model_artifact(
        self,
        model: Any,
        artifact_name: str,
        model_type: str = "pytorch",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save and log a model artifact with SHA-256 checksum.

        Args:
            model:         The model object (PyTorch nn.Module or sklearn/xgboost).
            artifact_name: Subdirectory name in the MLflow artifact store.
            model_type:    "pytorch" | "sklearn" | "xgboost"
            extra_metadata: Additional metadata to log as params.

        Returns:
            SHA-256 checksum of the serialised model.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            if model_type == "pytorch":
                import torch
                torch.save(model.state_dict(), model_path)
            else:
                import pickle
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            checksum = _sha256_file(model_path)

            # Write checksum sidecar
            checksum_path = Path(tmpdir) / "model.sha256"
            checksum_path.write_text(checksum)

            self.mlflow.log_artifacts(tmpdir, artifact_path=artifact_name)

        self.mlflow.log_param(f"{artifact_name}.sha256", checksum)
        if extra_metadata:
            self.log_params({f"{artifact_name}.{k}": v for k, v in extra_metadata.items()})

        logger.info("Model artifact logged: %s (sha256=%s...)", artifact_name, checksum[:16])
        return checksum

    # ------------------------------------------------------------------
    # PNG artifacts
    # ------------------------------------------------------------------

    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        horizon_label: str = "24h",
        artifact_path: str = "plots",
    ) -> None:
        """Generate and log a ROC curve PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {horizon_label} horizon")
            ax.legend(loc="lower right")
            fig.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, dpi=150)
                self.mlflow.log_artifact(f.name, artifact_path=artifact_path)
                os.unlink(f.name)
            plt.close(fig)
        except Exception as exc:
            logger.warning("ROC curve logging failed: %s", exc)

    def log_calibration_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        horizon_label: str = "24h",
        artifact_path: str = "plots",
    ) -> None:
        """Generate and log a calibration (reliability diagram) PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve

            fraction_of_positives, mean_predicted = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f"Calibration Plot — {horizon_label} horizon")
            ax.legend()
            fig.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, dpi=150)
                self.mlflow.log_artifact(f.name, artifact_path=artifact_path)
                os.unlink(f.name)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Calibration plot logging failed: %s", exc)

    # ------------------------------------------------------------------
    # CSV artifacts
    # ------------------------------------------------------------------

    def log_ablation_table(
        self,
        ablation_results: list[dict[str, Any]],
        artifact_path: str = "tables",
    ) -> None:
        """Log ablation comparison table as CSV."""
        self._log_csv(ablation_results, "ablation_comparison.csv", artifact_path)

    def log_subgroup_analysis(
        self,
        subgroup_results: list[dict[str, Any]],
        artifact_path: str = "tables",
    ) -> None:
        """Log subgroup analysis results as CSV."""
        self._log_csv(subgroup_results, "subgroup_analysis.csv", artifact_path)

    def log_nri_idi_table(
        self,
        nri_idi_results: dict[str, Any],
        artifact_path: str = "tables",
    ) -> None:
        """Log NRI/IDI results as CSV."""
        self._log_csv([nri_idi_results], "nri_idi.csv", artifact_path)

    def log_delong_results(
        self,
        delong_results: list[dict[str, Any]],
        artifact_path: str = "tables",
    ) -> None:
        """Log DeLong test results as CSV."""
        self._log_csv(delong_results, "delong_test.csv", artifact_path)

    def _log_csv(
        self,
        rows: list[dict[str, Any]],
        filename: str,
        artifact_path: str,
    ) -> None:
        """Write a list of dicts to CSV and log as MLflow artifact."""
        if not rows:
            return
        try:
            import csv

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, newline=""
            ) as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
                tmp_path = f.name

            self.mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            os.unlink(tmp_path)
        except Exception as exc:
            logger.warning("CSV artifact logging failed (%s): %s", filename, exc)

    # ------------------------------------------------------------------
    # Evaluation report
    # ------------------------------------------------------------------

    def log_evaluation_report(
        self,
        report_content: str,
        artifact_path: str = "reports",
    ) -> None:
        """Log evaluation_report.md as an MLflow artifact."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as f:
                f.write(report_content)
                tmp_path = f.name
            self.mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            os.unlink(tmp_path)
        except Exception as exc:
            logger.warning("Evaluation report logging failed: %s", exc)

    def generate_evaluation_report(
        self,
        model_name: str,
        metrics: dict[str, float],
        fold_metrics: list[dict[str, float]] | None = None,
        subgroup_results: list[dict[str, Any]] | None = None,
        nri_idi: dict[str, Any] | None = None,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a Markdown evaluation report string.

        Returns:
            Markdown string suitable for logging as evaluation_report.md.
        """
        lines: list[str] = [
            f"# Evaluation Report: {model_name}",
            "",
            "## Dataset",
        ]

        if dataset_metadata:
            lines += [
                f"- Samples: {dataset_metadata.get('n_samples', 'N/A')}",
                f"- Patients: {dataset_metadata.get('n_patients', 'N/A')}",
                f"- Features: {dataset_metadata.get('n_features', 'N/A')}",
                f"- Event rate (24h): {dataset_metadata.get('event_rate', 0):.1%}",
                f"- Feature version: {dataset_metadata.get('feature_version', 'N/A')}",
                "",
            ]

        lines += ["## Primary Metrics (Test Set)", ""]
        for k, v in sorted(metrics.items()):
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

        if fold_metrics:
            lines += ["## Cross-Validation Summary", ""]
            all_keys = set().union(*[m.keys() for m in fold_metrics])
            for key in sorted(all_keys):
                vals = [m[key] for m in fold_metrics if key in m]
                if vals:
                    lines.append(
                        f"- {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
                    )
            lines.append("")

        if nri_idi:
            lines += ["## NRI / IDI vs NEWS2 Baseline", ""]
            for k, v in nri_idi.items():
                lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
            lines.append("")

        if subgroup_results:
            lines += ["## Subgroup Analysis", ""]
            if subgroup_results:
                headers = list(subgroup_results[0].keys())
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in subgroup_results:
                    vals = [str(round(v, 4)) if isinstance(v, float) else str(v) for v in row.values()]
                    lines.append("| " + " | ".join(vals) + " |")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict with dot-separated keys."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, full_key))
        else:
            result[full_key] = v
    return result


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
