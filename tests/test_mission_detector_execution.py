"""Tests for detector-driven scenario execution (ScenarioExecutor)."""

from __future__ import annotations

from pathlib import Path

import pytest

from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    ManeuverRecord,
    OrbitStateRecord,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    ScenarioStateFile,
    StateFileClient,
    TimelineEventRecord,
    register_default_orekit_providers,
)
from astrodyn_core.mission.executor import (
    ManeuverFiredEvent,
    MissionExecutionReport,
    ScenarioExecutor,
)

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

CLIENT = StateFileClient()

_EPOCH_START = "2026-02-19T00:00:00Z"
_EPOCH_3H = "2026-02-19T03:00:00Z"
_EPOCH_6H = "2026-02-19T06:00:00Z"

_LEO_STATE = OrbitStateRecord(
    epoch=_EPOCH_START,
    frame="GCRF",
    representation="keplerian",
    elements={
        "a_m": 6_878_137.0,
        "e": 0.01,
        "i_deg": 51.6,
        "argp_deg": 45.0,
        "raan_deg": 120.0,
        "anomaly_deg": 0.0,
        "anomaly_type": "MEAN",
    },
    mu_m3_s2="WGS84",
    mass_kg=450.0,
)


def _build_numerical_propagator(state_record: OrbitStateRecord = _LEO_STATE):
    ctx = BuildContext.from_state_record(state_record)
    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        mass_kg=float(state_record.mass_kg or 450.0),
        integrator=IntegratorSpec(
            kind="dp54",
            min_step=1.0e-6,
            max_step=300.0,
            position_tolerance=1.0e-3,
        ),
    )
    builder = factory.build_builder(spec, ctx)
    return builder.buildPropagator(builder.getSelectedNormalizedParameters())


def _make_scenario(maneuvers: list[dict]) -> ScenarioStateFile:
    """Build a minimal scenario from raw maneuver dicts."""
    return ScenarioStateFile(
        initial_state=_LEO_STATE,
        maneuvers=tuple(ManeuverRecord.from_mapping(m) for m in maneuvers),
    )


# ---------------------------------------------------------------------------
# Unit tests: MissionExecutionReport
# ---------------------------------------------------------------------------

class TestMissionExecutionReport:
    def _sample_event(self, applied: bool) -> ManeuverFiredEvent:
        return ManeuverFiredEvent(
            maneuver_name="test-burn",
            epoch=_EPOCH_START,
            trigger_type="apogee",
            dv_inertial_mps=(10.0, 0.0, 0.0),
            applied=applied,
            guard_skip_reason=None if applied else "guard: sma too high",
        )

    def test_applied_events_filter(self):
        events = (self._sample_event(True), self._sample_event(False))
        report = MissionExecutionReport(
            events=events,
            total_dv_mps=10.0,
            propagation_start=_EPOCH_START,
            propagation_end=_EPOCH_3H,
        )
        assert len(report.applied_events()) == 1
        assert report.applied_events()[0].applied is True

    def test_skipped_events_filter(self):
        events = (self._sample_event(True), self._sample_event(False))
        report = MissionExecutionReport(
            events=events,
            total_dv_mps=10.0,
            propagation_start=_EPOCH_START,
            propagation_end=_EPOCH_3H,
        )
        assert len(report.skipped_events()) == 1
        assert report.skipped_events()[0].applied is False


# ---------------------------------------------------------------------------
# Integration tests: ScenarioExecutor
# ---------------------------------------------------------------------------

class TestScenarioExecutor:
    def test_executor_requires_propagator(self):
        scenario = _make_scenario([])
        with pytest.raises(TypeError, match="addEventDetector"):
            ScenarioExecutor("not-a-propagator", scenario)

    def test_executor_requires_scenario(self):
        propagator = _build_numerical_propagator()
        with pytest.raises(TypeError, match="ScenarioStateFile"):
            ScenarioExecutor(propagator, "not-a-scenario")

    def test_epoch_trigger_fires(self):
        """A DateDetector for a specific epoch should fire and apply the impulse."""
        maneuver = {
            "name": "epoch-burn",
            "trigger": {"type": "epoch", "epoch": "2026-02-19T00:30:00Z"},
            "model": {
                "type": "impulsive",
                "dv_mps": [10.0, 0.0, 0.0],
                "frame": "TNW",
            },
        }
        scenario = _make_scenario([maneuver])
        propagator = _build_numerical_propagator()
        executor = ScenarioExecutor(propagator, scenario)

        epoch_spec = OutputEpochSpec(
            start_epoch=_EPOCH_START,
            end_epoch="2026-02-19T01:00:00Z",
            step_seconds=1800.0,
        )
        series, report = executor.run_and_sample(epoch_spec, representation="keplerian")

        # The maneuver should have fired
        applied = report.applied_events()
        assert len(applied) == 1, f"Expected 1 applied event, got: {report.events}"
        assert applied[0].maneuver_name == "epoch-burn"
        assert applied[0].applied is True
        assert applied[0].dv_inertial_mps is not None
        assert len(series.states) == len(epoch_spec.epochs())

    def test_occurrence_first_fires_once(self):
        """occurrence='first' on an apogee trigger fires only on the first apogee."""
        maneuver = {
            "name": "one-time-apogee-burn",
            "trigger": {"type": "apogee", "occurrence": "first"},
            "model": {
                "type": "impulsive",
                "dv_mps": [5.0, 0.0, 0.0],
                "frame": "TNW",
            },
        }
        scenario = _make_scenario([maneuver])
        propagator = _build_numerical_propagator()
        executor = ScenarioExecutor(propagator, scenario)

        # 3-hour window: LEO period ~90 min, so ~2 apogees
        epoch_spec = OutputEpochSpec(
            start_epoch=_EPOCH_START,
            end_epoch=_EPOCH_3H,
            step_seconds=600.0,
        )
        _series, report = executor.run_and_sample(epoch_spec)

        applied = report.applied_events()
        assert len(applied) == 1, (
            f"occurrence='first' should fire exactly once, got {len(applied)} applied events"
        )

    def test_guard_skips_maneuver(self):
        """A guard that is already satisfied should cause the maneuver to be skipped."""
        # SMA ~6878 km; guard skips if sma > 5_000_000 m — always true
        maneuver = {
            "name": "guarded-burn",
            "trigger": {
                "type": "epoch",
                "epoch": "2026-02-19T00:30:00Z",
                "guard": {"sma_above_m": 5_000_000.0},
            },
            "model": {
                "type": "impulsive",
                "dv_mps": [10.0, 0.0, 0.0],
                "frame": "TNW",
            },
        }
        scenario = _make_scenario([maneuver])
        propagator = _build_numerical_propagator()
        executor = ScenarioExecutor(propagator, scenario)

        epoch_spec = OutputEpochSpec(
            start_epoch=_EPOCH_START,
            end_epoch="2026-02-19T01:00:00Z",
            step_seconds=1800.0,
        )
        _series, report = executor.run_and_sample(epoch_spec)

        assert len(report.applied_events()) == 0, "Guard should have prevented burn"
        assert len(report.skipped_events()) == 1
        assert report.skipped_events()[0].guard_skip_reason is not None
        assert "sma" in report.skipped_events()[0].guard_skip_reason

    def test_report_total_dv(self):
        """total_dv_mps should be the magnitude sum of applied burns."""
        maneuvers = [
            {
                "name": "burn-1",
                "trigger": {"type": "epoch", "epoch": "2026-02-19T00:20:00Z"},
                "model": {"type": "impulsive", "dv_mps": [10.0, 0.0, 0.0], "frame": "TNW"},
            },
            {
                "name": "burn-2",
                "trigger": {"type": "epoch", "epoch": "2026-02-19T00:40:00Z"},
                "model": {"type": "impulsive", "dv_mps": [0.0, 5.0, 0.0], "frame": "TNW"},
            },
        ]
        scenario = _make_scenario(maneuvers)
        propagator = _build_numerical_propagator()
        executor = ScenarioExecutor(propagator, scenario)

        epoch_spec = OutputEpochSpec(
            start_epoch=_EPOCH_START,
            end_epoch="2026-02-19T01:00:00Z",
            step_seconds=1200.0,
        )
        _series, report = executor.run_and_sample(epoch_spec)

        assert len(report.applied_events()) == 2
        # Each burn is ~10 m/s and ~5 m/s in TNW; inertial magnitudes will be close
        assert report.total_dv_mps > 5.0, f"Expected total_dv > 5 m/s, got {report.total_dv_mps}"

    def test_run_and_sample_representation(self):
        """run_and_sample should respect the representation parameter."""
        scenario = ScenarioStateFile(initial_state=_LEO_STATE)
        propagator = _build_numerical_propagator()
        executor = ScenarioExecutor(propagator, scenario)

        epoch_spec = OutputEpochSpec(
            start_epoch=_EPOCH_START,
            end_epoch="2026-02-19T00:30:00Z",
            step_seconds=900.0,
        )
        series, _report = executor.run_and_sample(epoch_spec, representation="keplerian")
        for record in series.states:
            assert record.representation == "keplerian"
            assert record.elements is not None
            assert "a_m" in record.elements

    def test_statefile_client_detector_mode(self):
        """StateFileClient.run_scenario_detector_mode should work end-to-end."""
        maneuver = {
            "name": "client-test-burn",
            "trigger": {"type": "epoch", "epoch": "2026-02-19T00:15:00Z"},
            "model": {"type": "impulsive", "dv_mps": [3.0, 0.0, 0.0], "frame": "TNW"},
        }
        scenario = _make_scenario([maneuver])
        propagator = _build_numerical_propagator()

        epoch_spec = OutputEpochSpec(
            start_epoch=_EPOCH_START,
            end_epoch="2026-02-19T00:30:00Z",
            step_seconds=900.0,
        )
        series, report = CLIENT.run_scenario_detector_mode(propagator, scenario, epoch_spec)

        assert isinstance(series.states[0], OrbitStateRecord)
        assert isinstance(report, MissionExecutionReport)
        assert len(report.applied_events()) == 1


# ---------------------------------------------------------------------------
# Integration: detector mode with existing scenario files
# ---------------------------------------------------------------------------

class TestDetectorModeWithScenarioFiles:
    def test_detector_mode_intent_raise_perigee(self):
        """Detector-mode with an intent maneuver from an existing scenario file."""
        scenario_path = Path("examples/state_files/leo_intent_mission.yaml")
        if not scenario_path.exists():
            pytest.skip("leo_intent_mission.yaml not found")

        scenario = CLIENT.load_state_file(scenario_path)
        assert scenario.initial_state is not None

        propagator = _build_numerical_propagator(scenario.initial_state)
        epoch_spec = OutputEpochSpec(
            start_epoch=scenario.initial_state.epoch,
            end_epoch="2026-02-19T02:00:00Z",
            step_seconds=600.0,
        )
        series, report = CLIENT.run_scenario_detector_mode(
            propagator, scenario, epoch_spec, representation="keplerian"
        )

        assert len(series.states) > 0
        assert isinstance(report, MissionExecutionReport)
        # Intent maneuver should have fired at apogee
        assert len(report.events) > 0
