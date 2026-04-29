"""
OncoIDT Training Service — FastAPI application.

Endpoints:
  POST /training/runs          — start a training run (async, returns run_id)
  GET  /training/runs/{run_id} — poll run status and metrics
  POST /training/ablation      — trigger full ablation suite
  GET  /training/models        — list model versions from MLflow registry
  POST /training/models/{version}/promote — flag model as production candidate

Requirements: 14.4, 14.5
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from shared.config import get_settings
from shared.db import close_engine
from shared.health import make_health_router
from shared.logging_config import configure_logging
from shared.metrics import graceful_shutdown, setup_metrics, start_error_rate_monitor

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.service_name, settings.log_level)

# ---------------------------------------------------------------------------
# In-memory run registry (production would use Redis or PostgreSQL)
# ---------------------------------------------------------------------------

_runs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CohortFilterRequest(BaseModel):
    start_date: datetime | None = None
    end_date: datetime | None = None
    diagnosis_codes: list[str] = Field(default_factory=list)
    chemo_regimens: list[str] = Field(default_factory=list)
    ward_ids: list[str] = Field(default_factory=list)


class TrainingRunRequest(BaseModel):
    feature_version: str = "v1"
    label_type: Literal["infection_event", "deterioration_event"] = "infection_event"
    horizons: list[int] = Field(default=[6, 12, 24, 48])
    cohort_filter: CohortFilterRequest = Field(default_factory=CohortFilterRequest)
    max_epochs: int = Field(default=50, ge=1, le=500)
    learning_rate: float = Field(default=1e-3, gt=0)
    hidden_dim: int = Field(default=256, ge=32)
    batch_size: int = Field(default=64, ge=8)
    device: str = "cpu"


class AblationRunRequest(BaseModel):
    feature_version: str = "v1"
    label_type: Literal["infection_event", "deterioration_event"] = "infection_event"
    cohort_filter: CohortFilterRequest = Field(default_factory=CohortFilterRequest)
    run_name_prefix: str = "ablation"
    max_epochs: int = Field(default=50, ge=1, le=500)
    device: str = "cpu"
    run_news2: bool = True
    run_lstm: bool = True
    run_xgboost: bool = True
    run_neural_cde: bool = True
    run_neural_cde_graph: bool = True


class PromoteRequest(BaseModel):
    approved_by: str = Field(..., min_length=1, description="Identity of the approving clinician/researcher")
    notes: str = ""


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    model_name: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    mlflow_run_id: str | None = None
    error: str | None = None


class ModelVersionInfo(BaseModel):
    name: str
    version: str
    stage: str
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str | None = None
    description: str | None = None


# ---------------------------------------------------------------------------
# Background training tasks
# ---------------------------------------------------------------------------

async def _run_training_task(
    run_id: str,
    request: TrainingRunRequest,
) -> None:
    """Background task: build dataset and train a single model run."""
    _runs[run_id]["status"] = "running"
    _runs[run_id]["started_at"] = datetime.now(timezone.utc)

    try:
        from services.training.dataset import CohortFilter, build_dataset
        from services.training.train import AblationConfig, _run_neural_cde_graph
        from shared.db import get_session_factory

        cohort = CohortFilter(
            start_date=request.cohort_filter.start_date,
            end_date=request.cohort_filter.end_date,
            diagnosis_codes=request.cohort_filter.diagnosis_codes,
            chemo_regimens=request.cohort_filter.chemo_regimens,
            ward_ids=request.cohort_filter.ward_ids,
        )

        factory = get_session_factory()
        async with factory() as session:
            dataset = await build_dataset(
                cohort_filter=cohort,
                feature_version=request.feature_version,
                label_type=request.label_type,
                horizons=request.horizons,
                db_session=session,
            )

        config = AblationConfig(
            experiment_name=settings.mlflow_experiment_name,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
            max_epochs=request.max_epochs,
            learning_rate=request.learning_rate,
            hidden_dim=request.hidden_dim,
            batch_size=request.batch_size,
            device=request.device,
            run_news2=False,
            run_lstm=False,
            run_xgboost=False,
            run_neural_cde=False,
            run_neural_cde_graph=True,
        )

        from services.training.mlflow_logger import OncoMLflowLogger
        mlflow_logger = OncoMLflowLogger(
            experiment_name=config.experiment_name,
            tracking_uri=config.mlflow_tracking_uri,
        )

        result = _run_neural_cde_graph(dataset, config, mlflow_logger)

        _runs[run_id]["status"] = "completed"
        _runs[run_id]["completed_at"] = datetime.now(timezone.utc)
        _runs[run_id]["metrics"] = result.metrics
        _runs[run_id]["mlflow_run_id"] = result.run_id
        _runs[run_id]["model_name"] = result.model_name

    except Exception as exc:
        logger.error("Training run %s failed: %s", run_id, exc, exc_info=True)
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["completed_at"] = datetime.now(timezone.utc)
        _runs[run_id]["error"] = str(exc)


async def _run_ablation_task(
    run_id: str,
    request: AblationRunRequest,
) -> None:
    """Background task: build dataset and run full ablation suite."""
    _runs[run_id]["status"] = "running"
    _runs[run_id]["started_at"] = datetime.now(timezone.utc)

    try:
        from services.training.dataset import CohortFilter, build_dataset
        from services.training.train import AblationConfig, train_ablation_suite
        from shared.db import get_session_factory

        cohort = CohortFilter(
            start_date=request.cohort_filter.start_date,
            end_date=request.cohort_filter.end_date,
            diagnosis_codes=request.cohort_filter.diagnosis_codes,
            chemo_regimens=request.cohort_filter.chemo_regimens,
            ward_ids=request.cohort_filter.ward_ids,
        )

        factory = get_session_factory()
        async with factory() as session:
            dataset = await build_dataset(
                cohort_filter=cohort,
                feature_version=request.feature_version,
                label_type=request.label_type,
                horizons=[6, 12, 24, 48],
                db_session=session,
            )

        config = AblationConfig(
            experiment_name=settings.mlflow_experiment_name,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
            run_name_prefix=request.run_name_prefix,
            device=request.device,
            max_epochs=request.max_epochs,
            run_news2=request.run_news2,
            run_lstm=request.run_lstm,
            run_xgboost=request.run_xgboost,
            run_neural_cde=request.run_neural_cde,
            run_neural_cde_graph=request.run_neural_cde_graph,
        )

        ablation_results = train_ablation_suite(config, dataset)

        best = max(
            (r for r in ablation_results.values() if r.metrics),
            key=lambda r: r.metrics.get("auroc", 0),
            default=None,
        )

        _runs[run_id]["status"] = "completed"
        _runs[run_id]["completed_at"] = datetime.now(timezone.utc)
        _runs[run_id]["metrics"] = best.metrics if best else {}
        _runs[run_id]["mlflow_run_id"] = best.run_id if best else None
        _runs[run_id]["model_name"] = "ablation_suite"

    except Exception as exc:
        logger.error("Ablation run %s failed: %s", run_id, exc, exc_info=True)
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["completed_at"] = datetime.now(timezone.utc)
        _runs[run_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    error_monitor_task = await start_error_rate_monitor(app, settings.service_name, settings.redis_url)
    logger.info("Training service started.")
    yield
    await graceful_shutdown([error_monitor_task])
    await close_engine()
    logger.info("Training service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoIDT Training Service",
    version=settings.service_version,
    lifespan=lifespan,
)
setup_metrics(app, settings.service_name)

app.include_router(make_health_router(service_name="training-service"))


# ---------------------------------------------------------------------------
# POST /training/runs
# ---------------------------------------------------------------------------

@app.post("/training/runs", response_model=RunStatusResponse, status_code=202)
async def start_training_run(
    request: TrainingRunRequest,
    background_tasks: BackgroundTasks,
) -> RunStatusResponse:
    """
    Start an async training run. Returns immediately with a run_id.
    Poll GET /training/runs/{run_id} for status and metrics.
    Requirement 14.4
    """
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "model_name": "NeuralCDE+Graph",
        "started_at": None,
        "completed_at": None,
        "metrics": {},
        "mlflow_run_id": None,
        "error": None,
    }
    background_tasks.add_task(_run_training_task, run_id, request)
    logger.info("Training run queued: %s", run_id)
    return RunStatusResponse(**_runs[run_id])


# ---------------------------------------------------------------------------
# GET /training/runs/{run_id}
# ---------------------------------------------------------------------------

@app.get("/training/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """
    Poll the status and metrics of a training run.
    Requirement 14.4
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return RunStatusResponse(**_runs[run_id])


# ---------------------------------------------------------------------------
# POST /training/ablation
# ---------------------------------------------------------------------------

@app.post("/training/ablation", response_model=RunStatusResponse, status_code=202)
async def start_ablation(
    request: AblationRunRequest,
    background_tasks: BackgroundTasks,
) -> RunStatusResponse:
    """
    Trigger the full ablation suite (NEWS2, LSTM, XGBoost, NeuralCDE, NeuralCDE+Graph).
    Returns immediately with a run_id. Poll GET /training/runs/{run_id} for status.
    Requirement 14.6
    """
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "model_name": "ablation_suite",
        "started_at": None,
        "completed_at": None,
        "metrics": {},
        "mlflow_run_id": None,
        "error": None,
    }
    background_tasks.add_task(_run_ablation_task, run_id, request)
    logger.info("Ablation suite queued: %s", run_id)
    return RunStatusResponse(**_runs[run_id])


# ---------------------------------------------------------------------------
# GET /training/models
# ---------------------------------------------------------------------------

@app.get("/training/models", response_model=list[ModelVersionInfo])
async def list_models() -> list[ModelVersionInfo]:
    """
    List all model versions from the MLflow model registry.
    Requirement 14.5
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(f"name='{settings.mlflow_model_registry_name}'")
        result: list[ModelVersionInfo] = []

        for v in versions:
            # Fetch metrics from the associated run
            metrics: dict[str, float] = {}
            try:
                run = client.get_run(v.run_id)
                metrics = {k: float(val) for k, val in run.data.metrics.items()}
            except Exception:
                pass

            result.append(ModelVersionInfo(
                name=v.name,
                version=str(v.version),
                stage=v.current_stage,
                run_id=v.run_id,
                metrics=metrics,
                created_at=str(v.creation_timestamp),
                description=v.description or "",
            ))

        return result

    except Exception as exc:
        logger.error("Failed to list models from MLflow: %s", exc)
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /training/models/{version}/promote
# ---------------------------------------------------------------------------

@app.post("/training/models/{version}/promote")
async def promote_model(
    version: str,
    request: PromoteRequest,
) -> dict[str, str]:
    """
    Flag a model version as a production candidate.
    Requires explicit human approval (approved_by field).
    The model is staged as 'Staging' — a separate deployment step
    moves it to 'Production' after clinical review.
    Requirement 14.4, 14.5
    """
    if not request.approved_by.strip():
        raise HTTPException(
            status_code=400,
            detail="Human approval required: 'approved_by' must identify the approving clinician.",
        )

    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=settings.mlflow_model_registry_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )

        client.update_model_version(
            name=settings.mlflow_model_registry_name,
            version=version,
            description=(
                f"Promotion candidate. Approved by: {request.approved_by}. "
                f"Notes: {request.notes}. "
                f"Promoted at: {datetime.now(timezone.utc).isoformat()}"
            ),
        )

        logger.info(
            "Model version %s promoted to Staging by %s", version, request.approved_by
        )
        return {
            "status": "promoted",
            "version": version,
            "stage": "Staging",
            "approved_by": request.approved_by,
            "message": "Model staged for production review. Requires final deployment approval.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Model promotion failed for version %s: %s", version, exc)
        raise HTTPException(status_code=503, detail=f"MLflow promotion failed: {exc}") from exc
