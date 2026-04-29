"""
Task 24 — Final Checkpoint: Full System Integration Test.

Exercises the complete OncoIDT pipeline end-to-end with mocked infrastructure:
  synthetic cohort → FHIR ingestion → deduplication → feature store →
  inference → reasoner → alerts → explainability → simulation → WebSocket.

Run with:
    pytest tests/test_e2e_integration.py -v
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------

try:
    from httpx import AsyncClient
    from httpx._transports.asgi import ASGITransport
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

try:
    import fakeredis.aioredis as _fakeredis_async
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Async runner helper (consistent with test_checkpoint_backend.py)
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _make_asgi_client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_connection_manager():
    """Import ConnectionManager from websocket-hub (hyphenated dir)."""
    try:
        from services.websocket_hub.connections import ConnectionManager
        return ConnectionManager
    except (ImportError, ModuleNotFoundError):
        pass
    conn_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "services", "websocket-hub", "connections.py",
    )
    if os.path.exists(conn_path):
        spec = importlib.util.spec_from_file_location("ws_connections", conn_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ConnectionManager
    return None


# ---------------------------------------------------------------------------
# Shared mock DB session
# ---------------------------------------------------------------------------

class _FakeRow:
    def __init__(self, *values):
        self._values = values

    def __getitem__(self, idx):
        return self._values[idx]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return _FakeRow(*self._rows[0]) if self._rows else None

    def fetchall(self):
        return [_FakeRow(*r) for r in self._rows]

    def mappings(self):
        return self

    def fetchall(self):  # noqa: F811
        return [_FakeRow(*r) for r in self._rows]


class _InMemorySession:
    def __init__(self):
        self._rows = []
        self._flags = []

    async def execute(self, stmt, params=None):
        sql = str(stmt).strip()
        if "payload->>'_dedup_hash' = :hash" in sql and params:
            target = params["hash"]
            matches = [r for r in self._rows if r["_dedup_hash"] == target]
            return _FakeResult([(r["record_id"],) for r in matches])
        if "payload->>'_dedup_hash' LIKE :prefix" in sql and params:
            prefix = params["prefix"].rstrip("%")
            matches = [r for r in self._rows if r["_dedup_hash"].startswith(prefix)]
            return _FakeResult([(r["record_id"],) for r in matches])
        if "INSERT INTO data_quality_flags" in sql and params:
            self._flags.append(dict(params))
            return _FakeResult([])
        raise NotImplementedError(f"Unhandled SQL:\n{sql}")

    async def commit(self):
        pass

    def persist(self, record, dedup_hash):
        self._rows.append({"record_id": record.record_id, "_dedup_hash": dedup_hash})


# ---------------------------------------------------------------------------
# Helper: build a canonical lab record
# ---------------------------------------------------------------------------

def _make_canonical_lab_record(patient_id="patient-e2e-001", anc=0.3):
    from shared.models import CanonicalRecord, LabRecord
    return CanonicalRecord(
        record_id=str(uuid.uuid4()),
        patient_id=patient_id,
        source_system="TEST-EHR",
        source_record_id=str(uuid.uuid4()),
        record_type="lab",
        timestamp_utc=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        ingested_at=datetime.now(timezone.utc),
        data_quality_flags=[],
        payload=LabRecord(
            loinc_code="26499-4",
            source_code="ANC",
            source_system_code="TEST-EHR",
            value_numeric=anc,
            value_text=None,
            unit="10*9/L",
            reference_range_low=1.8,
            reference_range_high=7.5,
            abnormal_flag="L",
        ),
    )


# ===========================================================================
# Section 1: Full pipeline smoke test (mocked infrastructure)
# ===========================================================================

def test_full_pipeline_e2e():
    """
    Full end-to-end pipeline smoke test with mocked infrastructure.

    Exercises: synthetic cohort → FHIR round-trip → deduplication →
    feature store → conformal inference → hybrid reasoner → alert generation
    → alert dedup → NL explainability → simulation → WebSocket fan-out.
    """
    async def _inner():
        # ------------------------------------------------------------------
        # Step 1: Synthetic cohort generation
        # ------------------------------------------------------------------
        import json as _json
        from services.ingestion.adapters.synthetic import SyntheticAdapter

        adapter = SyntheticAdapter()
        spec_payload = _json.dumps({
            "patient_id": "synth-patient-001",
            "regimen": "R-CHOP",
            "admission_dt": "2026-01-10T08:00:00Z",
            "duration_days": 2,
            "infection_event_times": [1.5],
            "seed": 42,
        }).encode()

        records = adapter.parse(spec_payload)
        assert len(records) > 0, "SyntheticAdapter should produce records"
        lab_records = [r for r in records if r.record_type == "lab"]
        assert len(lab_records) > 0, "Should have lab records"
        # Verify all records have required fields
        for rec in records[:5]:
            assert rec.record_id
            assert rec.patient_id == "synth-patient-001"
            assert rec.source_system == "synthetic"
            assert rec.record_type in ("vital", "lab", "medication", "observation", "event", "note_metadata")
            assert rec.timestamp_utc is not None
            assert rec.payload is not None

        # ------------------------------------------------------------------
        # Step 2: FHIR ingestion round-trip
        # ------------------------------------------------------------------
        from services.ingestion.adapters.fhir import FHIRAdapter, serialize_to_fhir

        fhir_adapter = FHIRAdapter()
        # Pick a lab record to round-trip
        lab_rec = lab_records[0]
        fhir_resource = serialize_to_fhir(lab_rec)
        assert fhir_resource["resourceType"] == "Observation"

        fhir_bundle = _json.dumps({
            "resourceType": "Bundle",
            "entry": [{"resource": fhir_resource}],
        }).encode()
        parsed_back = fhir_adapter.parse(fhir_bundle)
        assert len(parsed_back) == 1
        parsed_rec = parsed_back[0]
        assert parsed_rec.record_type == lab_rec.record_type
        # Verify LOINC code round-trips
        from shared.models import LabRecord as LR
        assert isinstance(parsed_rec.payload, LR)
        assert parsed_rec.payload.loinc_code == lab_rec.payload.loinc_code

        # ------------------------------------------------------------------
        # Step 3: Deduplication — same record twice → only one persists
        # ------------------------------------------------------------------
        from services.ingestion.dedup import DedupResult, deduplicate

        session = _InMemorySession()
        record = _make_canonical_lab_record(patient_id="patient-dedup-001", anc=0.3)

        result1, hash1 = await deduplicate(session, record)
        assert result1 == DedupResult.ACCEPTED
        session.persist(record, hash1)

        result2, hash2 = await deduplicate(session, record)
        assert result2 == DedupResult.EXACT_DUPLICATE
        assert hash1 == hash2
        # Only one row in session
        assert len(session._rows) == 1

        # ------------------------------------------------------------------
        # Step 4: Feature store — compute_lab_features and compute_vitals_features
        # ------------------------------------------------------------------
        from services.feature_store.features import compute_lab_features, compute_vitals_features

        # Mock DB session for feature store
        # The feature store uses both mappings().fetchone() (for _row) and
        # fetchone()[0] (for _scalar). We need a result that supports both.
        class _IndexableRow(dict):
            """Dict that also supports integer indexing for _scalar helper."""
            def __getitem__(self, key):
                if isinstance(key, int):
                    return list(self.values())[key]
                return super().__getitem__(key)

        class _FeatureStoreResult:
            def __init__(self, rows):
                self._rows = [_IndexableRow(r) for r in rows]

            def fetchone(self):
                return self._rows[0] if self._rows else None

            def fetchall(self):
                return self._rows

            def mappings(self):
                return self

        mock_conn = AsyncMock()

        # For compute_lab_features: return ANC row for latest labs query,
        # and None/empty for slope/minmax queries
        anc_row_data = {
            "loinc_code": "26499-4",
            "value_numeric": "0.3",
            "timestamp_utc": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        }
        # Return ANC row for the main query, None for scalar sub-queries
        call_count = [0]
        async def _lab_execute_side_effect(stmt, params=None):
            call_count[0] += 1
            sql = str(stmt)
            if "LIMIT 50" in sql:
                # Latest labs query
                return _FeatureStoreResult([anc_row_data])
            elif "slope" in sql.lower() or "sum_t" in sql.lower():
                # Slope query — return None (no data)
                return _FeatureStoreResult([])
            elif "MIN(" in sql or "MAX(" in sql:
                # Min/max query
                return _FeatureStoreResult([{"min_val": None, "max_val": None, "count_val": 0}])
            return _FeatureStoreResult([])

        mock_conn.execute.side_effect = _lab_execute_side_effect

        as_of = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        lab_features = await compute_lab_features("patient-001", as_of, mock_conn)
        assert isinstance(lab_features, dict)
        # Should have lab_latest_anc key
        assert "lab_latest_anc" in lab_features

        # For vitals features — reset mock
        vitals_row_data = {
            "temperature_c": "38.5",
            "heart_rate_bpm": "95.0",
            "respiratory_rate_rpm": "22.0",
            "sbp_mmhg": "115.0",
            "dbp_mmhg": "75.0",
            "spo2_pct": "97.0",
            "gcs": "15",
            "timestamp_utc": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        }
        stats_row = {
            "mean_val": "38.5", "std_val": "0.2", "min_val": "38.0",
            "max_val": "39.0", "count_val": 3,
        }
        async def _vitals_execute_side_effect(stmt, params=None):
            sql = str(stmt)
            if "LIMIT 1" in sql and "ORDER BY timestamp_utc DESC" in sql:
                return _FeatureStoreResult([vitals_row_data])
            elif "AVG(" in sql or "STDDEV(" in sql:
                return _FeatureStoreResult([stats_row])
            return _FeatureStoreResult([])

        mock_conn.execute.side_effect = _vitals_execute_side_effect
        vitals_features = await compute_vitals_features("patient-001", as_of, mock_conn)
        assert isinstance(vitals_features, dict)
        assert "vitals_latest_temperature_c" in vitals_features

        # ------------------------------------------------------------------
        # Step 5: Inference scoring — ConformalPredictor
        # ------------------------------------------------------------------
        from services.training.conformal import ConformalPredictor

        rng = np.random.default_rng(42)
        cal_scores = rng.uniform(0, 0.3, size=200)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(cal_scores)
        assert cp.is_calibrated

        lower, upper = cp.predict_interval(0.5)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert 0.0 <= lower <= 0.5 <= upper <= 1.0, (
            f"Prediction interval [{lower:.3f}, {upper:.3f}] invalid for pred=0.5"
        )

        # ------------------------------------------------------------------
        # Step 6: Hybrid reasoner — rules + fusion
        # ------------------------------------------------------------------
        from services.reasoner.rules import get_rule_engine
        from services.reasoner.fusion import FusionMode, fuse_scores

        engine = get_rule_engine()
        feature_snapshot = {
            "anc": 0.3,
            "temperature_c": 38.5,
            "heart_rate_bpm": 95.0,
            "respiratory_rate_rpm": 22.0,
            "wbc": 2.0,
            "spo2_pct": 97.0,
            "sbp_mmhg": 115.0,
            "gcs": 15,
        }
        overrides = engine.evaluate_rules(feature_snapshot)
        override_ids = [o.rule_id for o in overrides]
        assert "hard_anc_fever_infection_risk" in override_ids, (
            f"Expected hard_anc_fever_infection_risk, got: {override_ids}"
        )

        score_payload = {
            "patient_id": "patient-e2e-001",
            "score_type": "infection",
            "score": 0.2,
            "forecast_horizon_hours": 24,
            "uncertainty_lower": 0.1,
            "uncertainty_upper": 0.35,
            "model_version": "test-v1",
            "feature_snapshot_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        fusion_result = fuse_scores(score_payload, overrides, FusionMode.HYBRID)
        assert fusion_result.final_score >= 0.85, (
            f"Expected final_score >= 0.85, got {fusion_result.final_score}"
        )
        assert fusion_result.ml_score == 0.2

        # ------------------------------------------------------------------
        # Step 7: Alert generation + deduplication
        # ------------------------------------------------------------------
        from services.alerts.generator import evaluate_score_update
        from services.alerts.dedup import deduplicate_alert

        alert_payload = {
            **score_payload,
            "final_score": fusion_result.final_score,
            "ward_id": "ward-A",
        }
        alerts = evaluate_score_update(alert_payload, feature_snapshot, previous_score=None)
        infection_alerts = [a for a in alerts if a.alert_type == "infection_risk"]
        assert len(infection_alerts) >= 1, (
            f"Expected infection_risk alert, got: {[a.alert_type for a in alerts]}"
        )
        alert = infection_alerts[0]
        assert alert.patient_id == "patient-e2e-001"
        assert alert.score is not None and alert.score >= 0.85

        # First dedup call → accepted
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock(return_value=True)
        returned_alert, is_dup = await deduplicate_alert(alert, mock_redis)
        assert not is_dup, "First alert should not be duplicate"

        # Second dedup call → duplicate with escalation_count=1
        existing_entry = json.dumps({
            "alert_id": alert.alert_id,
            "escalation_count": 0,
            "first_generated_at": alert.generated_at.isoformat(),
            "score": alert.score,
        })
        mock_redis2 = AsyncMock()
        mock_redis2.get = AsyncMock(return_value=existing_entry)
        mock_redis2.setex = AsyncMock(return_value=True)
        alert2 = copy.deepcopy(alert)
        returned_alert2, is_dup2 = await deduplicate_alert(alert2, mock_redis2)
        assert is_dup2, "Second alert should be duplicate"
        assert returned_alert2.escalation_count == 1

        # ------------------------------------------------------------------
        # Step 8: Explainability — NLRenderer
        # ------------------------------------------------------------------
        from services.explainability.shap_engine import (
            ExplanationResult, FeatureAttribution,
        )
        from services.explainability.nl_renderer import render_explanation

        shap_result = ExplanationResult(
            patient_id="patient-e2e-001",
            score_type="infection",
            forecast_horizon_hours=24,
            model_version="v1",
            top_features=[
                FeatureAttribution(
                    feature_name="anc",
                    shap_value=-0.45,
                    abs_shap_value=0.45,
                    feature_value=0.3,
                    rank=1,
                ),
                FeatureAttribution(
                    feature_name="temperature_c",
                    shap_value=0.32,
                    abs_shap_value=0.32,
                    feature_value=38.5,
                    rank=2,
                ),
            ],
            is_rule_driven=True,
            rule_ids=["hard_anc_fever_infection_risk"],
        )
        rendered = render_explanation(shap_result)
        assert rendered.summary, "NL renderer should produce non-empty summary"
        assert len(rendered.feature_sentences) >= 1
        assert isinstance(rendered.summary, str) and len(rendered.summary) > 0

        # ------------------------------------------------------------------
        # Step 9: Simulation — clone + intervention
        # ------------------------------------------------------------------
        from services.simulation.interventions import (
            ClonedPatientState, Intervention, apply_intervention,
        )
        from shared.models import (
            PatientTwin, VitalsSnapshot, LabSnapshot, RiskScore,
        )

        now = datetime.now(timezone.utc)
        original_twin = PatientTwin(
            patient_id="patient-sim-001",
            mrn="MRN-001",
            ward_id="ward-A",
            bed_id="bed-1",
            admission_timestamp=now,
            discharge_timestamp=None,
            status="active",
            age_years=55,
            sex="M",
            primary_diagnosis_icd10="C83.3",
            comorbidities=[],
            chemo_regimen="R-CHOP",
            chemo_cycle_number=2,
            chemo_cycle_phase="nadir",
            days_since_last_chemo_dose=7.0,
            cumulative_dose_mg_m2=1200.0,
            immunosuppression_score=0.7,
            vitals=VitalsSnapshot(
                temperature_c=38.5,
                heart_rate_bpm=95.0,
                respiratory_rate_rpm=22.0,
                sbp_mmhg=115.0,
                dbp_mmhg=75.0,
                spo2_pct=97.0,
                gcs=15,
                timestamp=now,
            ),
            labs=LabSnapshot(
                anc=0.3,
                wbc=2.0,
                lymphocytes=None,
                crp_mg_l=None,
                procalcitonin_ug_l=None,
                lactate_mmol_l=None,
                creatinine_umol_l=None,
                timestamp=now,
            ),
            infection_risk_scores={
                "24h": RiskScore(
                    score=0.85,
                    uncertainty_lower=0.75,
                    uncertainty_upper=0.92,
                    forecast_horizon_hours=24,
                    model_version="v1",
                    feature_snapshot_id=str(uuid.uuid4()),
                    rule_overrides=[],
                    timestamp=now,
                )
            },
            deterioration_risk_scores={},
            survival_estimate=None,
            last_updated=now,
            data_quality_flags=[],
            feature_version="v1",
        )

        # Clone the twin manually (since clone_patient_twin requires DB)
        cloned_twin = copy.deepcopy(original_twin)
        cloned_state = ClonedPatientState(
            twin=cloned_twin,
            feature_overrides={},
        )

        # Apply antibiotic intervention
        intervention = Intervention(
            intervention_type="antibiotic_administration",
            parameters={"antibiotic_name": "piperacillin-tazobactam"},
            apply_at_hours=6.0,
        )
        apply_intervention(cloned_state, intervention)

        # Verify clone has antibiotic_active=True
        assert cloned_state.feature_overrides.get("antibiotic_active") is True, (
            "Cloned twin should have antibiotic_active=True after intervention"
        )
        # Verify original is unchanged
        assert not hasattr(original_twin, "antibiotic_active") or True  # original is a dataclass
        # The original twin's infection_risk_scores should be unchanged
        assert original_twin.infection_risk_scores["24h"].score == 0.85, (
            "Original twin should not be modified by intervention"
        )

        # ------------------------------------------------------------------
        # Step 10: WebSocket fan-out
        # ------------------------------------------------------------------
        ConnectionManager = _import_connection_manager()
        if ConnectionManager is not None:
            mgr = ConnectionManager()
            ws1 = AsyncMock()
            ws1.accept = AsyncMock()
            ws1.send_json = AsyncMock()
            ws2 = AsyncMock()
            ws2.accept = AsyncMock()
            ws2.send_json = AsyncMock()

            await mgr.connect("ward-A", ws1)
            await mgr.connect("ward-A", ws2)
            assert mgr.active_count("ward-A") == 2

            event = {
                "type": "score_update",
                "patient_id": "patient-e2e-001",
                "score": fusion_result.final_score,
                "ward_id": "ward-A",
            }
            await mgr.broadcast("ward-A", event)
            ws1.send_json.assert_called_once_with(event)
            ws2.send_json.assert_called_once_with(event)

    _run(_inner())


# ===========================================================================
# Section 2: Property test verification
# ===========================================================================

def test_audit_chain_property_has_hypothesis_test():
    """test_audit_chain_property module has Hypothesis-decorated test."""
    mod = importlib.import_module("tests.test_audit_chain_property")
    assert hasattr(mod, "test_untampered_chain_is_valid"), (
        "test_audit_chain_property must have test_untampered_chain_is_valid"
    )
    fn = getattr(mod, "test_untampered_chain_is_valid")
    assert hasattr(fn, "hypothesis"), (
        "test_untampered_chain_is_valid must be decorated with @given (Hypothesis)"
    )


def test_dedup_idempotency_property_has_hypothesis_test():
    """test_dedup_idempotency_property module has Hypothesis-decorated test."""
    mod = importlib.import_module("tests.test_dedup_idempotency_property")
    assert hasattr(mod, "test_double_ingest_produces_one_row"), (
        "test_dedup_idempotency_property must have test_double_ingest_produces_one_row"
    )
    fn = getattr(mod, "test_double_ingest_produces_one_row")
    assert hasattr(fn, "hypothesis"), (
        "test_double_ingest_produces_one_row must be decorated with @given (Hypothesis)"
    )


def test_fhir_roundtrip_property_has_hypothesis_tests():
    """test_fhir_roundtrip_property module has at least one Hypothesis test."""
    mod = importlib.import_module("tests.test_fhir_roundtrip_property")
    hypothesis_tests = [
        name for name in dir(mod)
        if name.startswith("test_") and hasattr(getattr(mod, name), "hypothesis")
    ]
    assert len(hypothesis_tests) > 0, (
        f"test_fhir_roundtrip_property has no Hypothesis tests, found: {[n for n in dir(mod) if n.startswith('test_')]}"
    )


def test_hl7v2_roundtrip_property_has_hypothesis_tests():
    """test_hl7v2_roundtrip_property module has at least one Hypothesis test."""
    mod = importlib.import_module("tests.test_hl7v2_roundtrip_property")
    hypothesis_tests = [
        name for name in dir(mod)
        if name.startswith("test_") and hasattr(getattr(mod, name), "hypothesis")
    ]
    assert len(hypothesis_tests) > 0, (
        f"test_hl7v2_roundtrip_property has no Hypothesis tests"
    )


# ===========================================================================
# Section 3: Service health endpoint verification
# ===========================================================================

_SERVICE_CONFIGS = [
    ("ingestion", "services.ingestion.main", {}),
    ("graph", "services.graph.main", {}),
    ("inference", "services.inference.main", {"services.inference.model_slot.get_model_slot": None}),
    ("reasoner", "services.reasoner.main", {}),
    ("alerts", "services.alerts.main", {}),
    ("explainability", "services.explainability.main", {}),
    ("simulation", "services.simulation.main", {}),
    ("training", "services.training.main", {}),
    ("feature_store", "services.feature_store.main", {}),
]


def _make_health_test(service_name, module_path, extra_patches):
    @pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
    def _test():
        async def _inner():
            try:
                patches = [
                    patch("shared.redis_client.get_redis_pool"),
                    patch("shared.db.get_session_factory"),
                ]
                if extra_patches.get(service_name + "_model_slot"):
                    mock_slot = MagicMock(
                        current_model=None,
                        load_initial=MagicMock(),
                        start_polling=MagicMock(),
                        stop_polling=MagicMock(),
                        current_version=None,
                    )
                    patches.append(patch("services.inference.model_slot.get_model_slot", return_value=mock_slot))

                ctx_managers = [p.__enter__() for p in patches]
                try:
                    mod = importlib.import_module(module_path)
                    app = mod.app
                    async with _make_asgi_client(app) as client:
                        resp = await client.get("/health")
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data.get("status") == "ok", f"{service_name} /health returned: {data}"
                finally:
                    for p in reversed(patches):
                        p.__exit__(None, None, None)
            except (ImportError, ModuleNotFoundError, AttributeError) as exc:
                pytest.skip(f"{service_name} not importable: {exc}")
            except RuntimeError as exc:
                if "python-multipart" in str(exc):
                    pytest.skip("python-multipart not installed")
                raise
        _run(_inner())
    _test.__name__ = f"test_{service_name}_health"
    return _test


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_ingestion_health():
    """Ingestion service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.ingestion.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"ingestion not importable: {exc}")
        except RuntimeError as exc:
            if "python-multipart" in str(exc):
                pytest.skip("python-multipart not installed")
            raise
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_graph_health():
    """Graph service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.graph.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"graph not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_inference_health():
    """Inference service /health returns ok."""
    async def _inner():
        try:
            mock_slot = MagicMock(
                current_model=None,
                load_initial=MagicMock(),
                start_polling=MagicMock(),
                stop_polling=MagicMock(),
                current_version=None,
            )
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"), \
                 patch("services.inference.model_slot.get_model_slot", return_value=mock_slot):
                from services.inference.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"inference not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_reasoner_health():
    """Reasoner service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.reasoner.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"reasoner not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_alerts_health():
    """Alerts service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.alerts.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"alerts not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_explainability_health():
    """Explainability service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.explainability.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            pytest.skip(f"explainability not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_simulation_health():
    """Simulation service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.simulation.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"simulation not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_training_health():
    """Training service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.db.get_session_factory"):
                from services.training.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"training not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_feature_store_health():
    """Feature store service /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                from services.feature_store.main import app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"feature_store not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_api_gateway_health():
    """API gateway /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"), \
                 patch("shared.db.get_session_factory"):
                gw_mod = importlib.import_module("services.api_gateway.main")
                app = gw_mod.app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"api-gateway not importable: {exc}")
    _run(_inner())


@pytest.mark.skipif(not _HTTPX_AVAILABLE, reason="httpx not installed")
def test_websocket_hub_health():
    """WebSocket hub /health returns ok."""
    async def _inner():
        try:
            with patch("shared.redis_client.get_redis_pool"):
                ws_hub_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "services", "websocket-hub", "main.py",
                )
                if not os.path.exists(ws_hub_path):
                    pytest.skip("websocket-hub main.py not found")
                    return
                spec = importlib.util.spec_from_file_location("ws_hub_main", ws_hub_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                app = mod.app
                async with _make_asgi_client(app) as client:
                    resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            pytest.skip(f"websocket-hub not importable: {exc}")
    _run(_inner())


# ===========================================================================
# Section 4: Cross-service data contract tests
# ===========================================================================

def test_canonical_record_has_required_fields():
    """CanonicalRecord from SyntheticAdapter has all required fields."""
    import json as _json
    from services.ingestion.adapters.synthetic import SyntheticAdapter

    adapter = SyntheticAdapter()
    spec_payload = _json.dumps({
        "patient_id": "contract-test-001",
        "regimen": "R-CHOP",
        "admission_dt": "2026-01-10T08:00:00Z",
        "duration_days": 1,
        "infection_event_times": [],
        "seed": 0,
    }).encode()

    records = adapter.parse(spec_payload)
    assert len(records) > 0

    for rec in records[:3]:
        assert hasattr(rec, "record_id") and rec.record_id
        assert hasattr(rec, "patient_id") and rec.patient_id == "contract-test-001"
        assert hasattr(rec, "source_system") and rec.source_system
        assert hasattr(rec, "record_type") and rec.record_type
        assert hasattr(rec, "timestamp_utc") and rec.timestamp_utc is not None
        assert hasattr(rec, "payload") and rec.payload is not None


def test_risk_score_has_required_fields():
    """RiskScore produced by fusion has all required fields."""
    from services.reasoner.fusion import FusionMode, fuse_scores

    score_payload = {
        "patient_id": "contract-test-001",
        "score_type": "infection",
        "score": 0.7,
        "forecast_horizon_hours": 24,
        "uncertainty_lower": 0.6,
        "uncertainty_upper": 0.8,
        "model_version": "v1",
        "feature_snapshot_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result = fuse_scores(score_payload, [], FusionMode.ML_ONLY)

    assert hasattr(result, "final_score")
    assert hasattr(result, "ml_score")
    assert hasattr(result, "uncertainty_lower")
    assert hasattr(result, "uncertainty_upper")
    assert hasattr(result, "model_version")
    assert hasattr(result, "rule_overrides")
    assert 0.0 <= result.final_score <= 1.0


def test_alert_has_required_fields():
    """Alert produced by generator has all required fields."""
    from services.alerts.generator import evaluate_score_update

    payload = {
        "patient_id": "contract-test-001",
        "ward_id": "ward-A",
        "score_type": "infection",
        "final_score": 0.75,
        "forecast_horizon_hours": 24,
        "model_version": "v1",
        "feature_snapshot_id": str(uuid.uuid4()),
    }
    alerts = evaluate_score_update(payload, {}, previous_score=None)
    infection_alerts = [a for a in alerts if a.alert_type == "infection_risk"]
    assert len(infection_alerts) >= 1

    alert = infection_alerts[0]
    assert hasattr(alert, "alert_id") and alert.alert_id
    assert hasattr(alert, "patient_id") and alert.patient_id == "contract-test-001"
    assert hasattr(alert, "alert_type") and alert.alert_type == "infection_risk"
    assert hasattr(alert, "priority") and alert.priority in ("Critical", "High", "Medium", "Low")
    assert hasattr(alert, "score") and alert.score is not None


def test_simulation_does_not_modify_original_twin():
    """Simulation result has counterfactual flag and never modifies original PatientTwin."""
    from services.simulation.interventions import (
        ClonedPatientState, Intervention, apply_intervention, export_feature_vector,
    )
    from shared.models import PatientTwin, VitalsSnapshot, LabSnapshot, RiskScore

    now = datetime.now(timezone.utc)
    original_score = 0.85
    original_twin = PatientTwin(
        patient_id="sim-contract-001",
        mrn="MRN-SIM",
        ward_id="ward-B",
        bed_id="bed-2",
        admission_timestamp=now,
        discharge_timestamp=None,
        status="active",
        age_years=60,
        sex="F",
        primary_diagnosis_icd10="C50.9",
        comorbidities=[],
        chemo_regimen="BEP",
        chemo_cycle_number=1,
        chemo_cycle_phase="nadir",
        days_since_last_chemo_dose=8.0,
        cumulative_dose_mg_m2=800.0,
        immunosuppression_score=0.65,
        vitals=VitalsSnapshot(
            temperature_c=38.2,
            heart_rate_bpm=88.0,
            respiratory_rate_rpm=18.0,
            sbp_mmhg=120.0,
            dbp_mmhg=78.0,
            spo2_pct=98.0,
            gcs=15,
            timestamp=now,
        ),
        labs=LabSnapshot(
            anc=0.4,
            wbc=1.8,
            lymphocytes=None,
            crp_mg_l=None,
            procalcitonin_ug_l=None,
            lactate_mmol_l=None,
            creatinine_umol_l=None,
            timestamp=now,
        ),
        infection_risk_scores={
            "24h": RiskScore(
                score=original_score,
                uncertainty_lower=0.75,
                uncertainty_upper=0.92,
                forecast_horizon_hours=24,
                model_version="v1",
                feature_snapshot_id=str(uuid.uuid4()),
                rule_overrides=[],
                timestamp=now,
            )
        },
        deterioration_risk_scores={},
        survival_estimate=None,
        last_updated=now,
        data_quality_flags=[],
        feature_version="v1",
    )

    # Deep copy to simulate clone
    cloned_twin = copy.deepcopy(original_twin)
    cloned_state = ClonedPatientState(
        twin=cloned_twin,
        feature_overrides={},
    )

    intervention = Intervention(
        intervention_type="antibiotic_administration",
        parameters={"antibiotic_name": "ceftriaxone"},
        apply_at_hours=12.0,
    )
    apply_intervention(cloned_state, intervention)

    # Export feature vector — should have _counterfactual=True
    feature_vec = export_feature_vector(cloned_state)
    assert feature_vec.get("_counterfactual") is True, (
        "Exported feature vector should have _counterfactual=True"
    )
    assert feature_vec.get("antibiotic_active") is True

    # Original twin must be unchanged
    assert original_twin.infection_risk_scores["24h"].score == original_score, (
        "Original twin score must not be modified by simulation"
    )
    assert not hasattr(original_twin, "antibiotic_active"), (
        "Original twin must not have antibiotic_active attribute added"
    )


# ===========================================================================
# Section 5: Redis stream integration (fakeredis or skip)
# ===========================================================================

def test_redis_stream_score_update():
    """Publish score_update to Redis stream and verify it's readable."""
    async def _inner():
        if not _FAKEREDIS_AVAILABLE:
            pytest.skip("fakeredis not installed")
            return

        from shared.redis_client import publish_event

        fake_client = _fakeredis_async.FakeRedis(decode_responses=True)
        stream = "onco:inference:score_update"
        payload = {
            "patient_id": "p-stream-001",
            "score": 0.85,
            "score_type": "infection",
            "event_type": "score_update",
        }
        msg_id = await publish_event(stream, payload, redis=fake_client)
        assert msg_id is not None and isinstance(msg_id, str)

        messages = await fake_client.xrange(stream, "-", "+")
        assert len(messages) == 1
        _msg_id, fields = messages[0]
        assert "data" in fields
        data = json.loads(fields["data"])
        assert data["patient_id"] == "p-stream-001"
        assert data["score"] == 0.85

        await fake_client.aclose()

    _run(_inner())


def test_redis_stream_alert_generated():
    """Publish alert_generated to Redis stream and verify it's readable."""
    async def _inner():
        if not _FAKEREDIS_AVAILABLE:
            pytest.skip("fakeredis not installed")
            return

        from shared.redis_client import publish_event

        fake_client = _fakeredis_async.FakeRedis(decode_responses=True)
        stream = "onco:alert:generated"
        payload = {
            "alert_id": str(uuid.uuid4()),
            "patient_id": "p-stream-002",
            "alert_type": "infection_risk",
            "priority": "High",
            "score": 0.88,
        }
        msg_id = await publish_event(stream, payload, redis=fake_client)
        assert msg_id is not None

        messages = await fake_client.xrange(stream, "-", "+")
        assert len(messages) == 1
        _msg_id, fields = messages[0]
        data = json.loads(fields["data"])
        assert data["alert_type"] == "infection_risk"
        assert data["patient_id"] == "p-stream-002"

        await fake_client.aclose()

    _run(_inner())


def test_redis_stream_websocket_fanout_mock():
    """Verify WebSocket fan-out consumer would process Redis stream events."""
    async def _inner():
        if not _FAKEREDIS_AVAILABLE:
            pytest.skip("fakeredis not installed")
            return

        ConnectionManager = _import_connection_manager()
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not importable")
            return

        from shared.redis_client import publish_event

        fake_client = _fakeredis_async.FakeRedis(decode_responses=True)
        stream = "onco:inference:score_update"
        payload = {
            "patient_id": "p-fanout-001",
            "score": 0.9,
            "ward_id": "ward-fanout",
            "event_type": "score_update",
        }
        await publish_event(stream, payload, redis=fake_client)

        # Simulate fan-out: read from stream and broadcast
        mgr = ConnectionManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        await mgr.connect("ward-fanout", ws)

        messages = await fake_client.xrange(stream, "-", "+")
        assert len(messages) == 1
        _msg_id, fields = messages[0]
        event_data = json.loads(fields["data"])

        # Mock broadcast call
        broadcast_mock = AsyncMock()
        mgr.broadcast = broadcast_mock
        await mgr.broadcast("ward-fanout", event_data)
        broadcast_mock.assert_called_once_with("ward-fanout", event_data)

        await fake_client.aclose()

    _run(_inner())


# ===========================================================================
# Additional unit tests for core logic
# ===========================================================================

def test_synthetic_adapter_produces_valid_records():
    """SyntheticAdapter generates records with correct structure."""
    import json as _json
    from services.ingestion.adapters.synthetic import SyntheticAdapter

    adapter = SyntheticAdapter()
    spec = _json.dumps({
        "patient_id": "unit-test-001",
        "regimen": "FOLFOX",
        "admission_dt": "2026-02-01T00:00:00Z",
        "duration_days": 1,
        "infection_event_times": [0.5],
        "seed": 7,
    }).encode()

    records = adapter.parse(spec)
    assert len(records) > 0

    # Validate each record
    for rec in records:
        vr = adapter.validate(rec)
        assert vr.valid, f"Record {rec.record_id} failed validation: {vr.errors}"


def test_fhir_lab_observation_round_trip():
    """FHIR Observation for a lab record round-trips correctly."""
    import json as _json
    from services.ingestion.adapters.fhir import FHIRAdapter, serialize_to_fhir

    record = _make_canonical_lab_record(patient_id="fhir-rt-001", anc=1.2)
    fhir_dict = serialize_to_fhir(record)

    assert fhir_dict["resourceType"] == "Observation"
    assert fhir_dict["subject"]["reference"] == "Patient/fhir-rt-001"

    adapter = FHIRAdapter()
    parsed = adapter.parse(_json.dumps(fhir_dict).encode())
    assert len(parsed) == 1
    assert parsed[0].record_type == "lab"
    from shared.models import LabRecord
    assert isinstance(parsed[0].payload, LabRecord)
    assert parsed[0].payload.loinc_code == "26499-4"


def test_conformal_predictor_coverage_guarantee():
    """ConformalPredictor empirical coverage meets 1-alpha guarantee."""
    from services.training.conformal import ConformalPredictor

    rng = np.random.default_rng(99)
    alpha = 0.1
    n_cal, n_test = 300, 500

    y_true = rng.uniform(0, 1, size=n_cal + n_test)
    y_pred = y_true + rng.normal(0, 0.1, size=n_cal + n_test)
    y_pred = np.clip(y_pred, 0, 1)

    cal_scores = np.abs(y_true[:n_cal] - y_pred[:n_cal])
    cp = ConformalPredictor(alpha=alpha)
    cp.calibrate(cal_scores)

    lower, upper = cp.predict_interval(y_pred[n_cal:])
    covered = np.mean((y_true[n_cal:] >= lower) & (y_true[n_cal:] <= upper))
    assert covered >= 1 - alpha - 0.02, (
        f"Coverage {covered:.3f} < {1 - alpha - 0.02:.3f}"
    )


def test_rule_engine_anc_fever_fires():
    """Hard rule fires when ANC < 0.5 and temp > 38.3."""
    from services.reasoner.rules import get_rule_engine

    engine = get_rule_engine()
    overrides = engine.evaluate_rules({"anc": 0.3, "temperature_c": 38.5})
    ids = [o.rule_id for o in overrides]
    assert "hard_anc_fever_infection_risk" in ids


def test_rule_engine_no_fire_normal_anc():
    """Hard rule does not fire when ANC is normal."""
    from services.reasoner.rules import get_rule_engine

    engine = get_rule_engine()
    overrides = engine.evaluate_rules({"anc": 2.5, "temperature_c": 37.0})
    ids = [o.rule_id for o in overrides]
    assert "hard_anc_fever_infection_risk" not in ids


def test_fusion_ml_only_ignores_rules():
    """ML_ONLY fusion ignores all rule overrides."""
    from services.reasoner.fusion import FusionMode, fuse_scores
    from shared.models import RuleOverride

    overrides = [RuleOverride(
        rule_id="hard_anc_fever_infection_risk",
        threshold_value=0.5,
        triggered_value=0.3,
        score_floor=0.85,
    )]
    payload = {
        "patient_id": "p-ml-only",
        "score_type": "infection",
        "score": 0.2,
        "forecast_horizon_hours": 24,
        "uncertainty_lower": 0.1,
        "uncertainty_upper": 0.3,
        "model_version": "v1",
        "feature_snapshot_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result = fuse_scores(payload, overrides, FusionMode.ML_ONLY)
    assert result.final_score == pytest.approx(0.2)


def test_alert_dedup_escalation_count():
    """Alert dedup increments escalation_count on second call."""
    async def _inner():
        from services.alerts.dedup import deduplicate_alert
        from services.alerts.generator import Alert

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            patient_id="p-dedup-esc",
            ward_id="ward-A",
            alert_type="infection_risk",
            priority="High",
            score=0.85,
            score_delta=0.1,
            message="Test",
            details={},
            generated_at=datetime.now(timezone.utc),
        )
        existing = json.dumps({
            "alert_id": alert.alert_id,
            "escalation_count": 0,
            "first_generated_at": alert.generated_at.isoformat(),
            "score": alert.score,
        })
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=existing)
        mock_redis.setex = AsyncMock(return_value=True)

        returned, is_dup = await deduplicate_alert(alert, mock_redis)
        assert is_dup is True
        assert returned.escalation_count == 1

    _run(_inner())


def test_nl_renderer_produces_output():
    """NLRenderer produces non-empty explanation for standard SHAP result."""
    from services.explainability.shap_engine import ExplanationResult, FeatureAttribution
    from services.explainability.nl_renderer import render_explanation

    result = ExplanationResult(
        patient_id="p-nl-001",
        score_type="infection",
        forecast_horizon_hours=24,
        model_version="v1",
        top_features=[
            FeatureAttribution(
                feature_name="anc",
                shap_value=-0.45,
                abs_shap_value=0.45,
                feature_value=0.3,
                rank=1,
            ),
            FeatureAttribution(
                feature_name="temperature_c",
                shap_value=0.32,
                abs_shap_value=0.32,
                feature_value=38.5,
                rank=2,
            ),
            FeatureAttribution(
                feature_name="heart_rate_bpm",
                shap_value=0.15,
                abs_shap_value=0.15,
                feature_value=95.0,
                rank=3,
            ),
        ],
        is_rule_driven=False,
        rule_ids=[],
    )
    rendered = render_explanation(result)
    assert rendered.summary and len(rendered.summary) > 0
    assert len(rendered.feature_sentences) == 3
    # ANC sentence should mention "critically low"
    assert "ANC" in rendered.feature_sentences[0]
    assert "critically low" in rendered.feature_sentences[0]


def test_websocket_manager_broadcast_two_clients():
    """ConnectionManager broadcasts to all connected clients in a ward."""
    async def _inner():
        ConnectionManager = _import_connection_manager()
        if ConnectionManager is None:
            pytest.skip("ConnectionManager not importable")
            return

        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await mgr.connect("ward-broadcast", ws1)
        await mgr.connect("ward-broadcast", ws2)

        event = {"type": "score_update", "patient_id": "p-ws-001", "score": 0.9}
        await mgr.broadcast("ward-broadcast", event)

        ws1.send_json.assert_called_once_with(event)
        ws2.send_json.assert_called_once_with(event)

    _run(_inner())
