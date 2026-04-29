"""
Checkpoint 11 — ML Stack smoke tests.

Verifies:
  1. Neural CDE forward/backward pass (skipped if torch/torchcde unavailable)
  2. GNN forward pass with heterogeneous edge types (skipped if torch/torch_geometric unavailable)
  3. Multi-task loss backward pass produces finite gradients (skipped if torch unavailable)
  4. Conformal predictor calibrate + predict_interval (pure numpy — always runs)
  5. DeepHit forward pass produces valid PMF (skipped if torch unavailable)
  6. Ablation suite smoke test on tiny synthetic dataset (skipped if torch/mlflow unavailable)

Requirements: 6.1, 6.2, 6.3, 6.5, 14.6
"""
from __future__ import annotations

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_onco_dataset(n_samples: int = 40, n_features: int = 16, seed: int = 0):
    """Build a minimal OncoDataset-like object without a DB."""
    from services.training.dataset import OncoDataset, DatasetSplit
    from datetime import datetime, timezone, timedelta

    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Binary labels: 4 infection horizons, 3 deterioration horizons
    labels_inf = (rng.random((n_samples, 4)) > 0.7).astype(np.float32)
    labels_det = (rng.random((n_samples, 3)) > 0.7).astype(np.float32)
    event_times = rng.uniform(1.0, 48.0, size=n_samples).astype(np.float32)
    event_observed = (rng.random(n_samples) > 0.3).astype(np.float32)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = np.array([base + timedelta(hours=i * 6) for i in range(n_samples)])
    patient_ids = np.array([f"p{i:03d}" for i in range(n_samples)])

    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    idx = np.arange(n_samples)
    split = DatasetSplit(
        train_indices=idx[:n_train],
        val_indices=idx[n_train:n_train + n_val],
        test_indices=idx[n_train + n_val:],
        train_cutoff=timestamps[n_train - 1],
        val_cutoff=timestamps[n_train + n_val - 1],
    )

    return OncoDataset(
        features=features,
        labels_infection=labels_inf,
        labels_deterioration=labels_det,
        event_times=event_times,
        event_observed=event_observed,
        patient_ids=patient_ids,
        timestamps=timestamps,
        feature_names=[f"feat_{i}" for i in range(n_features)],
        split=split,
        cv_folds=[],
        metadata={
            "n_samples": n_samples,
            "n_features": n_features,
            "n_patients": n_samples,
            "label_type": "infection_event",
            "feature_version": "v1",
            "event_rate": float(labels_inf[:, 2].mean()),
        },
    )


# ---------------------------------------------------------------------------
# 1. Neural CDE forward/backward pass
# ---------------------------------------------------------------------------

class TestNeuralCDEForwardBackward:
    """Req 6.1 — Neural CDE forward and backward passes."""

    def test_forward_pass_output_shapes(self):
        torch = pytest.importorskip("torch")
        torchcde = pytest.importorskip("torchcde")
        from services.training.models.neural_cde import NeuralCDE

        batch, seq_len, input_dim = 4, 5, 8
        hidden_dim = 32

        model = NeuralCDE(input_dim=input_dim, hidden_dim=hidden_dim)
        model.eval()

        x = torch.randn(batch, seq_len, input_dim)
        coeffs = torchcde.natural_cubic_coeffs(x)

        with torch.no_grad():
            inf_risk, det_risk, surv_params, repr_vec = model(coeffs)

        assert inf_risk.shape == (batch, 4), f"Expected (batch,4), got {inf_risk.shape}"
        assert det_risk.shape == (batch, 3), f"Expected (batch,3), got {det_risk.shape}"
        assert surv_params.shape == (batch, 2), f"Expected (batch,2), got {surv_params.shape}"
        assert repr_vec.shape == (batch, 256), f"Expected (batch,256), got {repr_vec.shape}"

    def test_forward_pass_output_ranges(self):
        torch = pytest.importorskip("torch")
        torchcde = pytest.importorskip("torchcde")
        from services.training.models.neural_cde import NeuralCDE

        batch, seq_len, input_dim = 4, 5, 8
        model = NeuralCDE(input_dim=input_dim, hidden_dim=32)
        model.eval()

        x = torch.randn(batch, seq_len, input_dim)
        coeffs = torchcde.natural_cubic_coeffs(x)

        with torch.no_grad():
            inf_risk, det_risk, surv_params, _ = model(coeffs)

        # Sigmoid outputs must be in [0, 1]
        assert inf_risk.min() >= 0.0 and inf_risk.max() <= 1.0
        assert det_risk.min() >= 0.0 and det_risk.max() <= 1.0
        # Softplus outputs must be > 0
        assert (surv_params > 0).all()

    def test_backward_pass_finite_gradients(self):
        torch = pytest.importorskip("torch")
        torchcde = pytest.importorskip("torchcde")
        from services.training.models.neural_cde import NeuralCDE
        from services.training.losses import MultiTaskLoss

        batch, seq_len, input_dim = 4, 5, 8
        model = NeuralCDE(input_dim=input_dim, hidden_dim=32)
        model.train()

        x = torch.randn(batch, seq_len, input_dim)
        coeffs = torchcde.natural_cubic_coeffs(x)

        inf_risk, det_risk, surv_params, _ = model(coeffs)

        criterion = MultiTaskLoss()
        inf_logits = torch.logit(inf_risk.clamp(1e-6, 1 - 1e-6))
        det_logits = torch.logit(det_risk.clamp(1e-6, 1 - 1e-6))
        inf_targets = torch.randint(0, 2, (batch, 4)).float()
        det_targets = torch.randint(0, 2, (batch, 3)).float()
        event_times = torch.rand(batch) * 48 + 1
        event_obs = torch.randint(0, 2, (batch,)).float()

        loss, _ = criterion(inf_logits, inf_targets, det_logits, det_targets,
                            surv_params, event_times, event_obs)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient in {name}"

    def test_irregular_time_series_variable_lengths(self):
        """CDE handles sequences of different lengths via spline interpolation."""
        torch = pytest.importorskip("torch")
        torchcde = pytest.importorskip("torchcde")
        from services.training.models.neural_cde import NeuralCDE

        input_dim = 6
        model = NeuralCDE(input_dim=input_dim, hidden_dim=32)
        model.eval()

        # Test with different sequence lengths
        for seq_len in [3, 7, 12]:
            x = torch.randn(2, seq_len, input_dim)
            coeffs = torchcde.natural_cubic_coeffs(x)
            with torch.no_grad():
                inf_risk, det_risk, surv_params, repr_vec = model(coeffs)
            assert inf_risk.shape == (2, 4)


# ---------------------------------------------------------------------------
# 2. GNN forward pass with heterogeneous edge types
# ---------------------------------------------------------------------------

class TestGNNForwardPass:
    """Req 6.2 — GNN forward pass with heterogeneous edge types."""

    def test_forward_pass_output_shape(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from services.training.models.gnn import GraphSAGE, NUM_EDGE_TYPES

        n_nodes, in_channels = 10, 16
        model = GraphSAGE(in_channels=in_channels, hidden_channels=64)
        model.eval()

        x = torch.randn(n_nodes, in_channels)
        # Create edges covering all edge types
        src = torch.tensor([0, 1, 2, 3, 4, 5])
        dst = torch.tensor([1, 2, 3, 4, 5, 6])
        edge_index = torch.stack([src, dst], dim=0)
        edge_type = torch.tensor([0, 1, 2, 3, 4, 5])  # one of each type

        with torch.no_grad():
            out = model(x, edge_index, edge_type)

        assert out.shape == (n_nodes, 64), f"Expected ({n_nodes}, 64), got {out.shape}"

    def test_forward_pass_with_single_edge_type(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from services.training.models.gnn import GraphSAGE

        n_nodes, in_channels = 8, 12
        model = GraphSAGE(in_channels=in_channels, hidden_channels=32)
        model.eval()

        x = torch.randn(n_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_type = torch.zeros(3, dtype=torch.long)  # all same type

        with torch.no_grad():
            out = model(x, edge_index, edge_type)

        assert out.shape == (n_nodes, 32)
        assert torch.isfinite(out).all()

    def test_forward_pass_no_edges(self):
        """GNN should handle graphs with no edges gracefully."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from services.training.models.gnn import GraphSAGE

        n_nodes, in_channels = 5, 8
        model = GraphSAGE(in_channels=in_channels, hidden_channels=32)
        model.eval()

        x = torch.randn(n_nodes, in_channels)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

        with torch.no_grad():
            out = model(x, edge_index, edge_type)

        assert out.shape == (n_nodes, 32)


# ---------------------------------------------------------------------------
# 3. Multi-task loss backward pass
# ---------------------------------------------------------------------------

class TestMultiTaskLoss:
    """Req 6.3 — Multi-task loss backward pass produces finite gradients."""

    def test_backward_produces_finite_gradients(self):
        torch = pytest.importorskip("torch")
        from services.training.losses import MultiTaskLoss

        batch = 8
        criterion = MultiTaskLoss(lambda_infection=1.0, lambda_deterioration=0.8,
                                  lambda_survival=0.3)

        inf_logits = torch.randn(batch, 4, requires_grad=True)
        det_logits = torch.randn(batch, 3, requires_grad=True)
        surv_params = torch.nn.functional.softplus(torch.randn(batch, 2))
        surv_params.retain_grad()

        inf_targets = torch.randint(0, 2, (batch, 4)).float()
        det_targets = torch.randint(0, 2, (batch, 3)).float()
        event_times = torch.rand(batch) * 47 + 1
        event_obs = torch.randint(0, 2, (batch,)).float()

        loss, components = criterion(inf_logits, inf_targets, det_logits, det_targets,
                                     surv_params, event_times, event_obs)
        loss.backward()

        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert torch.isfinite(inf_logits.grad).all(), "Non-finite grad in inf_logits"
        assert torch.isfinite(det_logits.grad).all(), "Non-finite grad in det_logits"

    def test_loss_components_are_positive(self):
        torch = pytest.importorskip("torch")
        from services.training.losses import MultiTaskLoss

        batch = 6
        criterion = MultiTaskLoss()
        inf_logits = torch.randn(batch, 4)
        det_logits = torch.randn(batch, 3)
        surv_params = torch.nn.functional.softplus(torch.randn(batch, 2))
        event_times = torch.rand(batch) * 47 + 1
        event_obs = torch.ones(batch)

        loss, components = criterion(inf_logits, torch.zeros(batch, 4),
                                     det_logits, torch.zeros(batch, 3),
                                     surv_params, event_times, event_obs)

        assert components["loss_infection"] >= 0
        assert components["loss_deterioration"] >= 0
        assert components["loss_survival"] >= 0
        assert components["loss_total"] >= 0

    def test_focal_loss_down_weights_easy_examples(self):
        """Focal loss should be lower for confident correct predictions."""
        torch = pytest.importorskip("torch")
        from services.training.losses import FocalLoss

        criterion = FocalLoss(gamma=2.0, alpha=0.75)

        # Very confident correct prediction (logit=5 → prob≈0.99, target=1)
        easy_logits = torch.tensor([5.0])
        easy_targets = torch.tensor([1.0])

        # Uncertain prediction (logit=0 → prob=0.5, target=1)
        hard_logits = torch.tensor([0.0])
        hard_targets = torch.tensor([1.0])

        easy_loss = criterion(easy_logits, easy_targets)
        hard_loss = criterion(hard_logits, hard_targets)

        assert easy_loss < hard_loss, \
            f"Focal loss should down-weight easy examples: easy={easy_loss:.4f}, hard={hard_loss:.4f}"


# ---------------------------------------------------------------------------
# 4. Conformal predictor (pure numpy — always runs)
# ---------------------------------------------------------------------------

class TestConformalPredictor:
    """Req 6.5 — Conformal predictor calibrate + predict_interval."""

    def test_calibrate_and_predict_interval_scalar(self):
        from services.training.conformal import ConformalPredictor

        rng = np.random.default_rng(42)
        cal_scores = rng.uniform(0, 0.5, size=200)

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(cal_scores)

        assert cp.is_calibrated
        lower, upper = cp.predict_interval(0.5)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= 0.5 <= upper

    def test_calibrate_and_predict_interval_array(self):
        from services.training.conformal import ConformalPredictor

        rng = np.random.default_rng(0)
        cal_scores = rng.uniform(0, 0.3, size=300)

        cp = ConformalPredictor(alpha=0.05)
        cp.calibrate(cal_scores)

        preds = np.array([0.2, 0.4, 0.6, 0.8])
        lower, upper = cp.predict_interval(preds)

        assert lower.shape == preds.shape
        assert upper.shape == preds.shape
        assert (lower <= preds).all()
        assert (upper >= preds).all()

    def test_interval_width_equals_twice_quantile(self):
        from services.training.conformal import ConformalPredictor

        rng = np.random.default_rng(7)
        cal_scores = rng.uniform(0, 0.4, size=500)

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(cal_scores)

        pred = 0.5
        lower, upper = cp.predict_interval(pred)
        q = cp.quantile

        assert abs((upper - lower) - 2 * q) < 1e-9, \
            f"Interval width {upper - lower:.6f} != 2*q {2*q:.6f}"

    def test_output_clamped_to_probability_range(self):
        from services.training.conformal import ConformalPredictor

        # Large calibration scores → wide intervals that would exceed [0,1]
        cal_scores = np.array([0.9] * 100)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(cal_scores)

        lower, upper = cp.predict_interval(0.5)
        assert lower >= 0.0
        assert upper <= 1.0

    def test_not_calibrated_raises_runtime_error(self):
        from services.training.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(0.5)

    def test_empty_calibration_set_raises(self):
        from services.training.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(ValueError, match="empty"):
            cp.calibrate(np.array([]))

    def test_empirical_coverage_meets_guarantee(self):
        """Marginal coverage on held-out set should be >= 1 - alpha."""
        from services.training.conformal import ConformalPredictor

        rng = np.random.default_rng(123)
        alpha = 0.1
        n_cal = 500
        n_test = 1000

        # True values and predictions from a simple model
        y_true = rng.uniform(0, 1, size=n_cal + n_test)
        y_pred = y_true + rng.normal(0, 0.1, size=n_cal + n_test)
        y_pred = np.clip(y_pred, 0, 1)

        cal_scores = np.abs(y_true[:n_cal] - y_pred[:n_cal])
        test_true = y_true[n_cal:]
        test_pred = y_pred[n_cal:]

        cp = ConformalPredictor(alpha=alpha)
        cp.calibrate(cal_scores)

        lower, upper = cp.predict_interval(test_pred)
        covered = np.mean((test_true >= lower) & (test_true <= upper))

        assert covered >= 1 - alpha - 0.02, \
            f"Coverage {covered:.3f} < {1 - alpha - 0.02:.3f} (1-alpha-eps)"


# ---------------------------------------------------------------------------
# 5. DeepHit forward pass
# ---------------------------------------------------------------------------

class TestDeepHitForwardPass:
    """Req 5.7 — DeepHit forward pass produces valid PMF."""

    def test_pmf_sums_to_one(self):
        torch = pytest.importorskip("torch")
        from services.training.models.deephit import DeepHit

        batch, in_features = 8, 32
        model = DeepHit(in_features=in_features, hidden_dim=64, num_bins=72)
        model.eval()

        x = torch.randn(batch, in_features)
        with torch.no_grad():
            pmf = model(x)

        sums = pmf.sum(dim=-1)
        assert pmf.shape == (batch, 72)
        assert torch.allclose(sums, torch.ones(batch), atol=1e-5), \
            f"PMF sums: {sums}"

    def test_pmf_values_non_negative(self):
        torch = pytest.importorskip("torch")
        from services.training.models.deephit import DeepHit

        model = DeepHit(in_features=16, hidden_dim=32, num_bins=72)
        model.eval()

        x = torch.randn(4, 16)
        with torch.no_grad():
            pmf = model(x)

        assert (pmf >= 0).all(), "PMF contains negative values"

    def test_predict_survival_output_keys(self):
        torch = pytest.importorskip("torch")
        from services.training.models.deephit import DeepHit

        batch, in_features = 4, 16
        model = DeepHit(in_features=in_features, hidden_dim=32, num_bins=72)
        model.eval()

        x = torch.randn(batch, in_features)
        with torch.no_grad():
            result = model.predict_survival(x)

        assert set(result.keys()) == {"pmf", "survival", "median_hours",
                                       "ci_80_lower", "ci_80_upper"}
        assert result["pmf"].shape == (batch, 72)
        assert result["survival"].shape == (batch, 72)
        assert result["median_hours"].shape == (batch,)

    def test_survival_function_is_decreasing(self):
        """S(t) = 1 - CDF(t) must be non-increasing."""
        torch = pytest.importorskip("torch")
        from services.training.models.deephit import DeepHit

        model = DeepHit(in_features=16, hidden_dim=32, num_bins=72)
        model.eval()

        x = torch.randn(4, 16)
        with torch.no_grad():
            result = model.predict_survival(x)

        survival = result["survival"]
        # Each row should be non-increasing
        diffs = survival[:, 1:] - survival[:, :-1]
        assert (diffs <= 1e-6).all(), "Survival function is not non-increasing"

    def test_median_within_time_range(self):
        torch = pytest.importorskip("torch")
        from services.training.models.deephit import DeepHit

        model = DeepHit(in_features=16, hidden_dim=32, num_bins=72)
        model.eval()

        x = torch.randn(8, 16)
        with torch.no_grad():
            result = model.predict_survival(x)

        # Bin centres range from 0.5 to 71.5
        assert (result["median_hours"] >= 0.5).all()
        assert (result["median_hours"] <= 71.5).all()


# ---------------------------------------------------------------------------
# 6. Ablation suite smoke test on tiny synthetic dataset
# ---------------------------------------------------------------------------

class TestAblationSuiteSmoke:
    """Req 14.6 — Ablation suite runs end-to-end on synthetic data."""

    def test_ablation_suite_news2_only(self):
        """NEWS2 baseline runs without torch/mlflow (uses random fallback)."""
        pytest.importorskip("torch")
        pytest.importorskip("mlflow")

        from services.training.train import AblationConfig, train_ablation_suite

        dataset = _make_synthetic_onco_dataset(n_samples=40, n_features=16)
        config = AblationConfig(
            experiment_name="test_checkpoint",
            mlflow_tracking_uri="",
            run_name_prefix="smoke",
            max_epochs=2,
            batch_size=8,
            hidden_dim=32,
            patience=2,
            run_news2=True,
            run_lstm=False,
            run_xgboost=False,
            run_neural_cde=False,
            run_neural_cde_graph=False,
        )

        results = train_ablation_suite(config, dataset)
        assert "NEWS2" in results
        assert isinstance(results["NEWS2"].metrics, dict)

    def test_ablation_suite_lstm_only(self):
        """LSTM baseline trains and evaluates on tiny synthetic data."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("mlflow")

        from services.training.train import AblationConfig, train_ablation_suite

        dataset = _make_synthetic_onco_dataset(n_samples=40, n_features=16)
        config = AblationConfig(
            experiment_name="test_checkpoint",
            mlflow_tracking_uri="",
            run_name_prefix="smoke",
            max_epochs=2,
            batch_size=8,
            hidden_dim=32,
            patience=2,
            run_news2=False,
            run_lstm=True,
            run_xgboost=False,
            run_neural_cde=False,
            run_neural_cde_graph=False,
        )

        results = train_ablation_suite(config, dataset)
        assert "LSTMBaseline" in results
        assert isinstance(results["LSTMBaseline"].metrics, dict)

    def test_ablation_suite_neural_cde_only(self):
        """Neural CDE trains and evaluates on tiny synthetic data."""
        pytest.importorskip("torch")
        pytest.importorskip("torchcde")
        pytest.importorskip("mlflow")

        from services.training.train import AblationConfig, train_ablation_suite

        dataset = _make_synthetic_onco_dataset(n_samples=40, n_features=16)
        config = AblationConfig(
            experiment_name="test_checkpoint",
            mlflow_tracking_uri="",
            run_name_prefix="smoke",
            max_epochs=2,
            batch_size=8,
            hidden_dim=32,
            patience=2,
            run_news2=False,
            run_lstm=False,
            run_xgboost=False,
            run_neural_cde=True,
            run_neural_cde_graph=False,
        )

        results = train_ablation_suite(config, dataset)
        assert "NeuralCDE" in results
        assert isinstance(results["NeuralCDE"].metrics, dict)

    def test_ablation_suite_full_pipeline(self):
        """Full ablation suite (all models) runs end-to-end."""
        pytest.importorskip("torch")
        pytest.importorskip("torchcde")
        pytest.importorskip("mlflow")
        pytest.importorskip("xgboost")

        from services.training.train import AblationConfig, train_ablation_suite

        dataset = _make_synthetic_onco_dataset(n_samples=60, n_features=16)
        config = AblationConfig(
            experiment_name="test_checkpoint",
            mlflow_tracking_uri="",
            run_name_prefix="smoke_full",
            max_epochs=2,
            batch_size=8,
            hidden_dim=32,
            patience=2,
            run_news2=True,
            run_lstm=True,
            run_xgboost=True,
            run_neural_cde=True,
            run_neural_cde_graph=True,
        )

        results = train_ablation_suite(config, dataset)
        expected_models = {"NEWS2", "LSTMBaseline", "XGBoostBaseline",
                           "NeuralCDE", "NeuralCDE+Graph"}
        assert set(results.keys()) == expected_models
        for name, result in results.items():
            assert isinstance(result.metrics, dict), f"{name} has no metrics dict"


# ---------------------------------------------------------------------------
# 7. Graph-Aware Transformer fusion layer
# ---------------------------------------------------------------------------

class TestGraphAwareTransformer:
    """Req 6.2 — Graph-Aware Transformer fusion layer."""

    def test_forward_pass_output_shape(self):
        torch = pytest.importorskip("torch")
        from services.training.models.graph_transformer import GraphAwareTransformer

        batch, num_neighbors = 4, 6
        cde_dim, gnn_dim, out_dim = 64, 64, 64

        model = GraphAwareTransformer(
            cde_dim=cde_dim, gnn_dim=gnn_dim, d_model=64,
            num_heads=4, out_dim=out_dim
        )
        model.eval()

        cde_hidden = torch.randn(batch, cde_dim)
        neighbor_emb = torch.randn(batch, num_neighbors, gnn_dim)
        edge_types = torch.randint(0, 6, (batch, num_neighbors))

        with torch.no_grad():
            out = model(cde_hidden, neighbor_emb, edge_types)

        assert out.shape == (batch, out_dim), f"Expected ({batch},{out_dim}), got {out.shape}"

    def test_forward_pass_without_edge_types(self):
        torch = pytest.importorskip("torch")
        from services.training.models.graph_transformer import GraphAwareTransformer

        batch, num_neighbors = 3, 4
        model = GraphAwareTransformer(cde_dim=32, gnn_dim=32, d_model=32,
                                      num_heads=4, out_dim=32)
        model.eval()

        cde_hidden = torch.randn(batch, 32)
        neighbor_emb = torch.randn(batch, num_neighbors, 32)

        with torch.no_grad():
            out = model(cde_hidden, neighbor_emb, neighbor_edge_types=None)

        assert out.shape == (batch, 32)
        assert torch.isfinite(out).all()
