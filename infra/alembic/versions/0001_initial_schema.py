"""Initial OncoIDT schema — all tables, hypertables, indexes, and vitals_hourly continuous aggregate.

Revision ID: 0001
Revises:
Create Date: 2026-03-23
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extensions
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ------------------------------------------------------------------
    # clinical_events  (hypertable)
    # ------------------------------------------------------------------
    op.create_table(
        "clinical_events",
        sa.Column("record_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("record_type", sa.Text(), nullable=False),
        sa.Column(
            "timestamp_utc",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("source_system", sa.Text(), nullable=False),
        sa.Column("source_record_id", sa.Text(), nullable=False),
        sa.Column("dedup_hash", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "data_quality_flags",
            postgresql.JSONB(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )
    op.execute(
        """
        SELECT create_hypertable(
            'clinical_events',
            'timestamp_utc',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        )
        """
    )
    op.create_index(
        "idx_clinical_events_patient_time",
        "clinical_events",
        ["patient_id", sa.text("timestamp_utc DESC")],
    )
    op.create_index(
        "idx_clinical_events_dedup",
        "clinical_events",
        ["dedup_hash"],
        unique=True,
    )

    # ------------------------------------------------------------------
    # risk_scores  (hypertable)
    # ------------------------------------------------------------------
    op.create_table(
        "risk_scores",
        sa.Column("score_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("score_type", sa.Text(), nullable=False),
        sa.Column("forecast_horizon_hours", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("uncertainty_lower", sa.Float(), nullable=True),
        sa.Column("uncertainty_upper", sa.Float(), nullable=True),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("feature_snapshot_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "rule_overrides",
            postgresql.JSONB(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "fusion_mode",
            sa.Text(),
            server_default=sa.text("'hybrid'"),
            nullable=False,
        ),
        sa.Column(
            "staleness_flag",
            sa.Boolean(),
            server_default=sa.text("FALSE"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.execute(
        """
        SELECT create_hypertable(
            'risk_scores',
            'timestamp',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        )
        """
    )
    op.create_index(
        "idx_risk_scores_patient_time",
        "risk_scores",
        ["patient_id", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_risk_scores_type_time",
        "risk_scores",
        ["score_type", sa.text("timestamp DESC")],
    )

    # ------------------------------------------------------------------
    # feature_snapshots  (hypertable)
    # ------------------------------------------------------------------
    op.create_table(
        "feature_snapshots",
        sa.Column("snapshot_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feature_version", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    op.execute(
        """
        SELECT create_hypertable(
            'feature_snapshots',
            'timestamp',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        )
        """
    )
    op.create_index(
        "idx_feature_snapshots_patient_time",
        "feature_snapshots",
        ["patient_id", sa.text("timestamp DESC")],
    )

    # ------------------------------------------------------------------
    # feature_store  (hypertable)
    # ------------------------------------------------------------------
    op.create_table(
        "feature_store",
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("feature_version", sa.Text(), nullable=False),
        sa.Column(
            "modality",
            sa.Text(),
            server_default=sa.text("'structured'"),
            nullable=False,
        ),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("lineage_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint("patient_id", "timestamp", "feature_version", "modality"),
    )
    op.execute(
        """
        SELECT create_hypertable(
            'feature_store',
            'timestamp',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        )
        """
    )
    op.create_index(
        "idx_feature_store_patient_version",
        "feature_store",
        ["patient_id", "feature_version", sa.text("timestamp DESC")],
    )

    # ------------------------------------------------------------------
    # feature_lineage
    # ------------------------------------------------------------------
    op.create_table(
        "feature_lineage",
        sa.Column(
            "lineage_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("feature_version", sa.Text(), nullable=False),
        sa.Column("modality", sa.Text(), nullable=False),
        sa.Column("source_fields", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("transformation_sql", sa.Text(), nullable=True),
        sa.Column("transformation_code_hash", sa.Text(), nullable=False),
        sa.Column(
            "deployed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("lineage_id"),
    )
    op.create_index(
        "idx_feature_lineage_version",
        "feature_lineage",
        ["feature_version", "modality"],
    )

    # ------------------------------------------------------------------
    # training_labels
    # ------------------------------------------------------------------
    op.create_table(
        "training_labels",
        sa.Column(
            "label_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("label_type", sa.Text(), nullable=False),
        sa.Column("event_timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("label_source", sa.Text(), nullable=False),
        sa.Column("horizon_6h", sa.Boolean(), nullable=True),
        sa.Column("horizon_12h", sa.Boolean(), nullable=True),
        sa.Column("horizon_24h", sa.Boolean(), nullable=True),
        sa.Column("horizon_48h", sa.Boolean(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("label_id"),
    )
    op.create_index(
        "idx_training_labels_patient",
        "training_labels",
        ["patient_id", sa.text("event_timestamp DESC")],
    )
    op.create_index(
        "idx_training_labels_type",
        "training_labels",
        ["label_type", sa.text("event_timestamp DESC")],
    )

    # ------------------------------------------------------------------
    # audit_log  (append-only, hash chain)
    # ------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("entry_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.Text(), nullable=False),
        sa.Column("resource_id", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("prev_hash", sa.Text(), nullable=False),
        sa.Column("entry_hash", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("entry_id"),
    )
    op.create_index(
        "idx_audit_log_user_time",
        "audit_log",
        ["user_id", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_audit_log_resource",
        "audit_log",
        ["resource_type", "resource_id", sa.text("timestamp DESC")],
    )

    # ------------------------------------------------------------------
    # alert_actions
    # ------------------------------------------------------------------
    op.create_table(
        "alert_actions",
        sa.Column(
            "action_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("snooze_until", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("action_id"),
    )
    op.create_index(
        "idx_alert_actions_alert",
        "alert_actions",
        ["alert_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_alert_actions_patient",
        "alert_actions",
        ["patient_id", sa.text("created_at DESC")],
    )

    # ------------------------------------------------------------------
    # simulation_sessions
    # ------------------------------------------------------------------
    op.create_table(
        "simulation_sessions",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("requesting_user_id", sa.Text(), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ward_id", sa.Text(), nullable=True),
        sa.Column(
            "simulation_type",
            sa.Text(),
            server_default=sa.text("'patient'"),
            nullable=False,
        ),
        sa.Column("input_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("interventions", postgresql.JSONB(), nullable=False),
        sa.Column("horizons", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("results", postgresql.JSONB(), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index(
        "idx_simulation_sessions_patient",
        "simulation_sessions",
        ["patient_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_simulation_sessions_user",
        "simulation_sessions",
        ["requesting_user_id", sa.text("created_at DESC")],
    )

    # ------------------------------------------------------------------
    # ward_explanations
    # ------------------------------------------------------------------
    op.create_table(
        "ward_explanations",
        sa.Column(
            "explanation_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("ward_id", sa.Text(), nullable=False),
        sa.Column(
            "computed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("patient_count", sa.Integer(), nullable=False),
        sa.Column("top_features", postgresql.JSONB(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("feature_version", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("explanation_id"),
    )
    op.create_index(
        "idx_ward_explanations_ward_time",
        "ward_explanations",
        ["ward_id", sa.text("computed_at DESC")],
    )

    # ------------------------------------------------------------------
    # graph_snapshots
    # ------------------------------------------------------------------
    op.create_table(
        "graph_snapshots",
        sa.Column(
            "snapshot_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("ward_id", sa.Text(), nullable=False),
        sa.Column("snapshot_timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("adjacency_json", postgresql.JSONB(), nullable=False),
        sa.Column("node_count", sa.Integer(), nullable=False),
        sa.Column("edge_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    op.create_index(
        "idx_graph_snapshots_ward_time",
        "graph_snapshots",
        ["ward_id", sa.text("snapshot_timestamp DESC")],
    )

    # ------------------------------------------------------------------
    # alert_performance_log
    # ------------------------------------------------------------------
    op.create_table(
        "alert_performance_log",
        sa.Column(
            "log_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alert_type", sa.Text(), nullable=False),
        sa.Column("priority", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("acknowledged_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("action_latency_s", sa.Float(), nullable=True),
        sa.Column("clinician_feedback", sa.Text(), nullable=True),
        sa.Column("shift_id", sa.Text(), nullable=True),
        sa.Column("ward_id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("log_id"),
    )
    op.create_index(
        "idx_alert_perf_ward_time",
        "alert_performance_log",
        ["ward_id", sa.text("generated_at DESC")],
    )

    # ------------------------------------------------------------------
    # vitals_hourly — TimescaleDB continuous aggregate
    # Must be created AFTER clinical_events hypertable exists.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS vitals_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            patient_id,
            time_bucket('1 hour', timestamp_utc)            AS bucket,
            avg((payload->>'value_numeric')::float)         AS avg_value,
            min((payload->>'value_numeric')::float)         AS min_value,
            max((payload->>'value_numeric')::float)         AS max_value,
            count(*)                                        AS observation_count
        FROM clinical_events
        WHERE record_type = 'vital'
        GROUP BY patient_id, bucket
        WITH NO DATA
        """
    )
    op.execute(
        """
        SELECT add_continuous_aggregate_policy(
            'vitals_hourly',
            start_offset => INTERVAL '2 hours',
            end_offset   => INTERVAL '1 minute',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
        """
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.execute("DROP MATERIALIZED VIEW IF EXISTS vitals_hourly CASCADE")

    for table in [
        "alert_performance_log",
        "graph_snapshots",
        "ward_explanations",
        "simulation_sessions",
        "alert_actions",
        "audit_log",
        "training_labels",
        "feature_lineage",
        "feature_store",
        "feature_snapshots",
        "risk_scores",
        "clinical_events",
    ]:
        op.drop_table(table)
