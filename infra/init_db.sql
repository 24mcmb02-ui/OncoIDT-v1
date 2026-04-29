-- =============================================================================
-- OncoIDT PostgreSQL + TimescaleDB schema initialisation
-- Run once against a fresh database that already has the TimescaleDB extension.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- clinical_events  (hypertable, 1-week chunks)
-- Stores all normalised CanonicalRecord payloads.
-- =============================================================================
CREATE TABLE IF NOT EXISTS clinical_events (
    record_id           UUID            NOT NULL,
    patient_id          UUID            NOT NULL,
    record_type         TEXT            NOT NULL,   -- vital | lab | medication | observation | event | note_metadata
    timestamp_utc       TIMESTAMPTZ     NOT NULL,
    ingested_at         TIMESTAMPTZ     NOT NULL    DEFAULT NOW(),
    source_system       TEXT            NOT NULL,
    source_record_id    TEXT            NOT NULL,
    dedup_hash          TEXT            NOT NULL,   -- SHA-256(patient_id||source_system||timestamp_utc||record_type)
    payload             JSONB           NOT NULL,
    data_quality_flags  JSONB           NOT NULL    DEFAULT '[]'
);

SELECT create_hypertable(
    'clinical_events',
    'timestamp_utc',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_clinical_events_patient_time
    ON clinical_events (patient_id, timestamp_utc DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_clinical_events_dedup
    ON clinical_events (dedup_hash, timestamp_utc);

-- =============================================================================
-- risk_scores  (hypertable, 1-week chunks)
-- =============================================================================
CREATE TABLE IF NOT EXISTS risk_scores (
    score_id                UUID        NOT NULL,
    patient_id              UUID        NOT NULL,
    score_type              TEXT        NOT NULL,   -- infection | deterioration
    forecast_horizon_hours  INT         NOT NULL,
    score                   FLOAT       NOT NULL,
    uncertainty_lower       FLOAT,
    uncertainty_upper       FLOAT,
    model_version           TEXT        NOT NULL,
    feature_snapshot_id     UUID        NOT NULL,
    rule_overrides          JSONB       NOT NULL    DEFAULT '[]',
    fusion_mode             TEXT        NOT NULL    DEFAULT 'hybrid',
    staleness_flag          BOOLEAN     NOT NULL    DEFAULT FALSE,
    timestamp               TIMESTAMPTZ NOT NULL
);

SELECT create_hypertable(
    'risk_scores',
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_risk_scores_patient_time
    ON risk_scores (patient_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_risk_scores_type_time
    ON risk_scores (score_type, timestamp DESC);

-- =============================================================================
-- feature_snapshots  (hypertable, 1-week chunks)
-- Point-in-time correct feature vectors used for inference.
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id     UUID        NOT NULL,
    patient_id      UUID        NOT NULL,
    feature_version TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    features        JSONB       NOT NULL
);

SELECT create_hypertable(
    'feature_snapshots',
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_snapshots_pk
    ON feature_snapshots (snapshot_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_patient_time
    ON feature_snapshots (patient_id, timestamp DESC);

-- =============================================================================
-- feature_store  (hypertable, composite PK)
-- Versioned feature store with modality support.
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_store (
    patient_id      UUID        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    feature_version TEXT        NOT NULL,
    modality        TEXT        NOT NULL    DEFAULT 'structured',  -- structured | text | imaging | sensor
    features        JSONB       NOT NULL,
    lineage_id      UUID        NOT NULL,
    PRIMARY KEY (patient_id, timestamp, feature_version, modality)
);

SELECT create_hypertable(
    'feature_store',
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_feature_store_patient_version
    ON feature_store (patient_id, feature_version, timestamp DESC);

-- =============================================================================
-- feature_lineage
-- Tracks transformation provenance for each feature version.
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_lineage (
    lineage_id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_version         TEXT        NOT NULL,
    modality                TEXT        NOT NULL,
    source_fields           TEXT[]      NOT NULL,
    transformation_sql      TEXT,
    transformation_code_hash TEXT       NOT NULL,
    deployed_at             TIMESTAMPTZ NOT NULL    DEFAULT NOW(),
    description             TEXT
);

CREATE INDEX IF NOT EXISTS idx_feature_lineage_version
    ON feature_lineage (feature_version, modality);

-- =============================================================================
-- training_labels
-- Ground-truth event labels for supervised training.
-- =============================================================================
CREATE TABLE IF NOT EXISTS training_labels (
    label_id        UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID        NOT NULL,
    label_type      TEXT        NOT NULL,   -- infection_event | deterioration_event | icu_transfer
    event_timestamp TIMESTAMPTZ NOT NULL,
    label_source    TEXT        NOT NULL,   -- clinical_confirmed | synthetic | retrospective_icd
    horizon_6h      BOOLEAN,
    horizon_12h     BOOLEAN,
    horizon_24h     BOOLEAN,
    horizon_48h     BOOLEAN,
    metadata        JSONB       NOT NULL    DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_training_labels_patient
    ON training_labels (patient_id, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_training_labels_type
    ON training_labels (label_type, event_timestamp DESC);

-- =============================================================================
-- audit_log
-- Append-only tamper-evident log with SHA-256 hash chain.
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id        BIGSERIAL   PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL    DEFAULT NOW(),
    user_id         TEXT        NOT NULL,
    action          TEXT        NOT NULL,
    resource_type   TEXT        NOT NULL,
    resource_id     TEXT        NOT NULL,
    details         JSONB,
    prev_hash       TEXT        NOT NULL,
    entry_hash      TEXT        NOT NULL    -- SHA-256(prev_hash||timestamp||user_id||action||resource_id||details)
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_time
    ON audit_log (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_resource
    ON audit_log (resource_type, resource_id, timestamp DESC);

-- =============================================================================
-- alert_actions
-- Records acknowledge / snooze / escalate actions on alerts.
-- =============================================================================
CREATE TABLE IF NOT EXISTS alert_actions (
    action_id       UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id        UUID        NOT NULL,
    patient_id      UUID        NOT NULL,
    user_id         TEXT        NOT NULL,
    action_type     TEXT        NOT NULL,   -- acknowledge | snooze | escalate
    reason          TEXT,
    snooze_until    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alert_actions_alert
    ON alert_actions (alert_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_alert_actions_patient
    ON alert_actions (patient_id, created_at DESC);

-- =============================================================================
-- simulation_sessions
-- Stores counterfactual simulation inputs and outputs.
-- =============================================================================
CREATE TABLE IF NOT EXISTS simulation_sessions (
    session_id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    requesting_user_id  TEXT        NOT NULL,
    patient_id          UUID        NOT NULL,
    ward_id             TEXT,
    simulation_type     TEXT        NOT NULL    DEFAULT 'patient',  -- patient | ward
    input_snapshot      JSONB       NOT NULL,
    interventions       JSONB       NOT NULL,
    horizons            INT[]       NOT NULL,
    results             JSONB,
    status              TEXT        NOT NULL    DEFAULT 'pending',  -- pending | running | completed | failed
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL    DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_simulation_sessions_patient
    ON simulation_sessions (patient_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_simulation_sessions_user
    ON simulation_sessions (requesting_user_id, created_at DESC);

-- =============================================================================
-- ward_explanations
-- Aggregated global SHAP explanations per ward, updated every 30 minutes.
-- =============================================================================
CREATE TABLE IF NOT EXISTS ward_explanations (
    explanation_id  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    ward_id         TEXT        NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL    DEFAULT NOW(),
    patient_count   INT         NOT NULL,
    top_features    JSONB       NOT NULL,   -- [{feature, mean_abs_shap, direction}]
    model_version   TEXT        NOT NULL,
    feature_version TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ward_explanations_ward_time
    ON ward_explanations (ward_id, computed_at DESC);

-- =============================================================================
-- graph_snapshots
-- Serialised adjacency list snapshots of the Neo4j graph for training.
-- =============================================================================
CREATE TABLE IF NOT EXISTS graph_snapshots (
    snapshot_id     UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    ward_id         TEXT        NOT NULL,
    snapshot_timestamp TIMESTAMPTZ NOT NULL,
    adjacency_json  JSONB       NOT NULL,
    node_count      INT         NOT NULL,
    edge_count      INT         NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_graph_snapshots_ward_time
    ON graph_snapshots (ward_id, snapshot_timestamp DESC);

-- =============================================================================
-- alert_performance_log
-- Tracks alert-to-action latency and volume for quality improvement.
-- =============================================================================
CREATE TABLE IF NOT EXISTS alert_performance_log (
    log_id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id            UUID        NOT NULL,
    patient_id          UUID        NOT NULL,
    alert_type          TEXT        NOT NULL,
    priority            TEXT        NOT NULL,
    generated_at        TIMESTAMPTZ NOT NULL,
    acknowledged_at     TIMESTAMPTZ,
    action_latency_s    FLOAT,
    clinician_feedback  TEXT,       -- false_positive | true_positive | not_assessed
    shift_id            TEXT,
    ward_id             TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alert_perf_ward_time
    ON alert_performance_log (ward_id, generated_at DESC);

-- =============================================================================
-- Continuous aggregate: vitals_hourly
-- Pre-aggregated hourly vitals for fast dashboard queries.
-- =============================================================================
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
GROUP BY patient_id, time_bucket('1 hour', timestamp_utc)
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'vitals_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);
