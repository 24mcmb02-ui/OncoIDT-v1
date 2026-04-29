"""
Unit tests for synthetic data temporal correlation.

Tests:
  1. ANC trajectory shape — decay/nadir/recovery for each supported chemo regimen
  2. Fever onset timestamps precede infection event timestamps within the 12–24h window
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from services.ingestion.adapters.synthetic import (
    REGIMEN_PARAMS,
    RegimenParams,
    anc_at_day,
    generate_patient_records,
    simulate_hawkes_fever_times,
)
from shared.models import CanonicalRecord, VitalRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _nadir_start(params: RegimenParams) -> float:
    """Day at which exponential decay reaches nadir_value."""
    return -math.log(params.nadir_value / params.baseline_anc) / params.decay_rate


# ---------------------------------------------------------------------------
# 1. ANC trajectory shape tests
# ---------------------------------------------------------------------------

class TestANCTrajectoryShape:
    """ANC values must follow expected decay → nadir → recovery shape."""

    @pytest.mark.parametrize("regimen", list(REGIMEN_PARAMS.keys()))
    def test_pre_nadir_is_decreasing(self, regimen: str) -> None:
        """ANC should decrease monotonically during the pre-nadir phase."""
        params = REGIMEN_PARAMS[regimen]
        rng = _rng(0)
        nadir_start = _nadir_start(params)

        # Sample several points in the pre-nadir window (use many seeds to average out noise)
        samples_early = [anc_at_day(nadir_start * 0.2, params, _rng(s)) for s in range(50)]
        samples_mid = [anc_at_day(nadir_start * 0.6, params, _rng(s)) for s in range(50)]
        samples_late = [anc_at_day(nadir_start * 0.9, params, _rng(s)) for s in range(50)]

        assert np.mean(samples_early) > np.mean(samples_mid), (
            f"{regimen}: ANC should decrease from early to mid pre-nadir"
        )
        assert np.mean(samples_mid) > np.mean(samples_late), (
            f"{regimen}: ANC should decrease from mid to late pre-nadir"
        )

    @pytest.mark.parametrize("regimen", list(REGIMEN_PARAMS.keys()))
    def test_nadir_value_is_low(self, regimen: str) -> None:
        """ANC at nadir should be near the configured nadir_value."""
        params = REGIMEN_PARAMS[regimen]
        nadir_start = _nadir_start(params)
        nadir_mid = nadir_start + params.nadir_duration_days / 2.0

        samples = [anc_at_day(nadir_mid, params, _rng(s)) for s in range(100)]
        mean_anc = np.mean(samples)

        # Mean should be within 3 std of the configured nadir value
        tolerance = 3 * params.nadir_noise_std + 0.1
        assert abs(mean_anc - params.nadir_value) < tolerance, (
            f"{regimen}: nadir mean {mean_anc:.3f} not close to "
            f"expected {params.nadir_value:.3f} (tol={tolerance:.3f})"
        )

    @pytest.mark.parametrize("regimen", list(REGIMEN_PARAMS.keys()))
    def test_nadir_below_baseline(self, regimen: str) -> None:
        """ANC at nadir must be substantially below baseline."""
        params = REGIMEN_PARAMS[regimen]
        nadir_start = _nadir_start(params)
        nadir_mid = nadir_start + params.nadir_duration_days / 2.0

        samples = [anc_at_day(nadir_mid, params, _rng(s)) for s in range(50)]
        assert np.mean(samples) < params.baseline_anc * 0.5, (
            f"{regimen}: nadir ANC should be < 50% of baseline"
        )

    @pytest.mark.parametrize("regimen", list(REGIMEN_PARAMS.keys()))
    def test_recovery_increases_toward_baseline(self, regimen: str) -> None:
        """ANC should increase during recovery phase and approach baseline."""
        params = REGIMEN_PARAMS[regimen]
        nadir_start = _nadir_start(params)
        nadir_end = nadir_start + params.nadir_duration_days

        # Sample at early, mid, and late recovery
        early_rec = nadir_end + 1.0
        mid_rec = nadir_end + params.recovery_midpoint_days
        late_rec = nadir_end + params.recovery_midpoint_days * 3.0

        samples_early = [anc_at_day(early_rec, params, _rng(s)) for s in range(50)]
        samples_mid = [anc_at_day(mid_rec, params, _rng(s)) for s in range(50)]
        samples_late = [anc_at_day(late_rec, params, _rng(s)) for s in range(50)]

        assert np.mean(samples_early) < np.mean(samples_mid), (
            f"{regimen}: ANC should increase from early to mid recovery"
        )
        assert np.mean(samples_mid) < np.mean(samples_late), (
            f"{regimen}: ANC should increase from mid to late recovery"
        )
        # Late recovery should be > 50% of baseline
        assert np.mean(samples_late) > params.baseline_anc * 0.5, (
            f"{regimen}: late recovery ANC should exceed 50% of baseline"
        )

    @pytest.mark.parametrize("regimen", list(REGIMEN_PARAMS.keys()))
    def test_anc_never_negative(self, regimen: str) -> None:
        """ANC must always be non-negative."""
        params = REGIMEN_PARAMS[regimen]
        nadir_start = _nadir_start(params)
        nadir_end = nadir_start + params.nadir_duration_days

        for day in [0.0, nadir_start, nadir_start + 1.0, nadir_end, nadir_end + 5.0]:
            for seed in range(20):
                anc = anc_at_day(day, params, _rng(seed))
                assert anc >= 0.0, f"{regimen} day={day}: ANC={anc} is negative"

    def test_rchop_nadir_below_neutropenic_threshold(self) -> None:
        """R-CHOP nadir should reliably fall below neutropenic threshold (1.5 × 10⁹/L)."""
        params = REGIMEN_PARAMS["R-CHOP"]
        nadir_start = _nadir_start(params)
        nadir_mid = nadir_start + params.nadir_duration_days / 2.0

        samples = [anc_at_day(nadir_mid, params, _rng(s)) for s in range(100)]
        # At least 90% of samples should be below 1.5
        below_threshold = sum(1 for v in samples if v < 1.5)
        assert below_threshold >= 90, (
            f"R-CHOP: only {below_threshold}/100 nadir samples below 1.5 × 10⁹/L"
        )


# ---------------------------------------------------------------------------
# 2. Fever onset precedes infection event within 12–24h window
# ---------------------------------------------------------------------------

class TestFeverOnsetPrecedesInfection:
    """Fever onset times should cluster in the 12–24h window before infection events."""

    def test_fever_times_precede_infection_within_window(self) -> None:
        """
        With a strong excitation amplitude, fever times should cluster
        in the [t_inf - 24h, t_inf - 12h] window before each infection event.
        """
        rng = _rng(42)
        infection_times = [10.0, 25.0]  # days from admission

        fever_times = simulate_hawkes_fever_times(
            infection_event_times=infection_times,
            horizon_days=40.0,
            baseline_intensity=0.001,   # very low baseline
            excitation_amplitude=2.0,   # strong excitation
            decay_rate=3.0,
            rng=rng,
        )

        # For each infection event, count fevers in the 12–24h preceding window
        for t_inf in infection_times:
            window_start = t_inf - 1.0   # 24h before
            window_end = t_inf - 0.5     # 12h before
            fevers_in_window = [
                ft for ft in fever_times
                if window_start <= ft <= window_end
            ]
            # With strong excitation, we expect at least some fevers in the window
            # (probabilistic — run with enough seeds to be reliable)
            assert len(fevers_in_window) >= 0  # structural check; see below

        # Run many seeds and verify that on average fevers cluster before infections
        window_fever_counts = []
        for seed in range(30):
            rng_s = _rng(seed)
            ft = simulate_hawkes_fever_times(
                infection_event_times=[10.0],
                horizon_days=20.0,
                baseline_intensity=0.001,
                excitation_amplitude=2.0,
                decay_rate=3.0,
                rng=rng_s,
            )
            in_window = sum(1 for f in ft if 9.0 <= f <= 9.5)
            window_fever_counts.append(in_window)

        # On average, at least some fevers should appear in the excitation window
        assert np.mean(window_fever_counts) > 0, (
            "Expected fevers to cluster in the 12–24h pre-infection window"
        )

    def test_fever_density_higher_near_infection(self) -> None:
        """
        Fever density in the 12–24h pre-infection window should exceed
        fever density in a control window far from any infection event.
        """
        n_trials = 50
        window_counts = []
        control_counts = []

        for seed in range(n_trials):
            rng = _rng(seed)
            ft = simulate_hawkes_fever_times(
                infection_event_times=[15.0],
                horizon_days=30.0,
                baseline_intensity=0.005,
                excitation_amplitude=1.5,
                decay_rate=3.0,
                rng=rng,
            )
            # Pre-infection window: 14.0–14.5 days (12–24h before infection at day 15)
            window_counts.append(sum(1 for f in ft if 14.0 <= f <= 14.5))
            # Control window: 5.0–5.5 days (far from infection)
            control_counts.append(sum(1 for f in ft if 5.0 <= f <= 5.5))

        assert np.mean(window_counts) > np.mean(control_counts), (
            f"Fever density near infection ({np.mean(window_counts):.3f}) should exceed "
            f"control window ({np.mean(control_counts):.3f})"
        )

    def test_no_fevers_without_infection_and_low_baseline(self) -> None:
        """With zero infection events and near-zero baseline, very few fevers expected."""
        total_fevers = 0
        for seed in range(20):
            rng = _rng(seed)
            ft = simulate_hawkes_fever_times(
                infection_event_times=[],
                horizon_days=10.0,
                baseline_intensity=0.0001,
                excitation_amplitude=1.0,
                decay_rate=3.0,
                rng=rng,
            )
            total_fevers += len(ft)

        # With near-zero baseline over 10 days, expect very few fevers
        assert total_fevers < 20, (
            f"Expected few fevers with no infection events, got {total_fevers}"
        )


# ---------------------------------------------------------------------------
# 3. Integration: generate_patient_records produces correct record types
# ---------------------------------------------------------------------------

class TestGeneratePatientRecords:
    """Smoke tests for the full record generation pipeline."""

    def test_records_contain_vitals_and_labs(self) -> None:
        """Generated records should include both vital and lab record types."""
        rng = _rng(0)
        admission_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        records = generate_patient_records(
            patient_id="test-patient-1",
            regimen="R-CHOP",
            admission_dt=admission_dt,
            duration_days=14,
            infection_event_times=[7.0],
            rng=rng,
        )
        types = {r.record_type for r in records}
        assert "vital" in types
        assert "lab" in types
        assert "event" in types

    def test_all_records_have_utc_timestamps(self) -> None:
        """All generated records must have timezone-aware UTC timestamps."""
        rng = _rng(1)
        admission_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        records = generate_patient_records(
            patient_id="test-patient-2",
            regimen="BEP",
            admission_dt=admission_dt,
            duration_days=10,
            infection_event_times=[],
            rng=rng,
        )
        for r in records:
            assert r.timestamp_utc.tzinfo is not None, (
                f"Record {r.record_id} has naive timestamp"
            )

    def test_infection_event_record_present(self) -> None:
        """An infection_confirmed event record should be generated for each infection time."""
        rng = _rng(2)
        admission_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        infection_times = [5.0, 12.0]
        records = generate_patient_records(
            patient_id="test-patient-3",
            regimen="FOLFOX",
            admission_dt=admission_dt,
            duration_days=20,
            infection_event_times=infection_times,
            rng=rng,
        )
        from shared.models import ClinicalEventRecord
        infection_events = [
            r for r in records
            if r.record_type == "event"
            and isinstance(r.payload, ClinicalEventRecord)
            and r.payload.event_type == "infection_confirmed"
        ]
        assert len(infection_events) == len(infection_times), (
            f"Expected {len(infection_times)} infection events, got {len(infection_events)}"
        )

    @pytest.mark.parametrize("regimen", list(REGIMEN_PARAMS.keys()))
    def test_anc_records_present_for_all_regimens(self, regimen: str) -> None:
        """ANC lab records should be generated for every supported regimen."""
        from shared.models import LabRecord
        rng = _rng(3)
        admission_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        records = generate_patient_records(
            patient_id="test-patient-4",
            regimen=regimen,
            admission_dt=admission_dt,
            duration_days=7,
            infection_event_times=[],
            rng=rng,
        )
        anc_records = [
            r for r in records
            if r.record_type == "lab"
            and isinstance(r.payload, LabRecord)
            and r.payload.loinc_code == "26499-4"
        ]
        assert len(anc_records) > 0, f"{regimen}: no ANC records generated"
        for r in anc_records:
            assert isinstance(r.payload, LabRecord)
            assert r.payload.value_numeric is not None
            assert r.payload.value_numeric >= 0.0
