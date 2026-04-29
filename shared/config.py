"""
OncoIDT centralised settings — pydantic-settings v2.
All services import `get_settings()` to access configuration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Service ports
    # ------------------------------------------------------------------
    api_gateway_port: int = 8000
    ingestion_service_port: int = 8001
    graph_service_port: int = 8002
    inference_service_port: int = 8003
    reasoner_service_port: int = 8004
    alert_service_port: int = 8005
    explainability_service_port: int = 8006
    simulation_service_port: int = 8007
    training_service_port: int = 8008
    feature_store_service_port: int = 8009
    websocket_hub_port: int = 8010

    # ------------------------------------------------------------------
    # Service base URLs (used for inter-service calls)
    # ------------------------------------------------------------------
    ingestion_service_url: str = "http://ingestion-service:8001"
    graph_service_url: str = "http://graph-service:8002"
    inference_service_url: str = "http://inference-service:8003"
    reasoner_service_url: str = "http://reasoner-service:8004"
    alert_service_url: str = "http://alert-service:8005"
    explainability_service_url: str = "http://explainability-service:8006"
    simulation_service_url: str = "http://simulation-service:8007"
    training_service_url: str = "http://training-service:8008"
    feature_store_service_url: str = "http://feature-store-service:8009"
    websocket_hub_url: str = "http://websocket-hub:8010"

    # ------------------------------------------------------------------
    # PostgreSQL / TimescaleDB
    # ------------------------------------------------------------------
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "oncoidt"
    postgres_user: str = "oncoidt"
    postgres_password: str = "oncoidt_secret"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_secret"

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    redis_stream_max_len: int = 100_000   # MAXLEN for XADD trimming

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "oncoidt"
    mlflow_model_registry_name: str = "oncoidt-model"

    # ------------------------------------------------------------------
    # JWT / Auth
    # ------------------------------------------------------------------
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # ------------------------------------------------------------------
    # mTLS certificate paths
    # ------------------------------------------------------------------
    mtls_ca_cert_path: str = "/certs/ca.crt"
    mtls_service_cert_path: str = "/certs/service.crt"
    mtls_service_key_path: str = "/certs/service.key"

    # ------------------------------------------------------------------
    # Feature store
    # ------------------------------------------------------------------
    feature_store_default_version: str = "v1"
    feature_store_sla_ms: int = 100

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    inference_batch_rescore_interval_seconds: int = 300   # 5 minutes
    inference_model_poll_interval_seconds: int = 30
    inference_model_retain_seconds: int = 300             # 5 minutes after swap

    # ------------------------------------------------------------------
    # Alert thresholds (defaults, overridable via rules.yaml)
    # ------------------------------------------------------------------
    alert_infection_risk_threshold: float = 0.6
    alert_deterioration_risk_threshold: float = 0.65
    alert_anc_critical_threshold: float = 0.5            # × 10⁹/L
    alert_dedup_window_minutes: int = 30

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    prometheus_port: int = 9090
    grafana_port: int = 3001
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Service identity (set per-service via env var)
    # ------------------------------------------------------------------
    service_name: str = "oncoidt"
    service_version: str = "0.1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
