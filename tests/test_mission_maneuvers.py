from __future__ import annotations

from pathlib import Path

import pytest

from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    ManeuverRecord,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    ScenarioStateFile,
    StateFileClient,
    register_default_orekit_providers,
)
from astrodyn_core.states.validation import parse_epoch_utc

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.propagation import SpacecraftState  # noqa: E402

CLIENT = StateFileClient()


def _build_numerical_from_scenario(scenario_path: Path):
    scenario = CLIENT.load_state_file(scenario_path)
    if scenario.initial_state is None:
        raise RuntimeError("Scenario needs initial_state for propagation tests.")

    ctx = BuildContext.from_state_record(
        scenario.initial_state,
        universe=scenario.universe,
        metadata={"source_scenario": scenario_path.name},
    )
    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        mass_kg=float(scenario.initial_state.mass_kg or 450.0),
        integrator=IntegratorSpec(
            kind="dp54",
            min_step=1.0e-6,
            max_step=300.0,
            position_tolerance=1.0e-3,
        ),
    )
    builder = factory.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    return scenario, propagator


def _perigee_m(record) -> float:
    elements = record.elements or {}
    return float(elements["a_m"]) * (1.0 - float(elements["e"]))


def _a_m(record) -> float:
    elements = record.elements or {}
    return float(elements["a_m"])


def test_compile_intent_maneuvers_keplerian() -> None:
    scenario_path = Path("examples/state_files/leo_intent_mission.yaml")
    scenario = CLIENT.load_state_file(scenario_path)
    assert scenario.initial_state is not None

    ctx = BuildContext.from_state_record(scenario.initial_state, universe=scenario.universe)
    initial_mass = float(scenario.initial_state.mass_kg or 450.0)
    initial_state = SpacecraftState(ctx.initial_orbit, initial_mass)

    compiled = CLIENT.compile_scenario_maneuvers(scenario, initial_state)
    assert len(compiled) == 2
    assert compiled[0].trigger_type == "apogee"
    assert compiled[1].trigger_type == "ascending_node"

    dv0 = compiled[0].dv_inertial_mps
    dv1 = compiled[1].dv_inertial_mps
    norm0 = (dv0[0] ** 2 + dv0[1] ** 2 + dv0[2] ** 2) ** 0.5
    norm1 = (dv1[0] ** 2 + dv1[1] ** 2 + dv1[2] ** 2) ** 0.5
    assert norm0 > 0.0
    assert norm1 > 0.0


def test_increment_targets_supported_for_raise_intents() -> None:
    seed = CLIENT.load_state_file(Path("examples/state_files/leo_intent_mission.yaml"))
    assert seed.initial_state is not None
    scenario = ScenarioStateFile(
        initial_state=seed.initial_state,
        maneuvers=(
            ManeuverRecord(
                name="raise-rp-inc",
                trigger={"type": "apogee"},
                model={
                    "type": "intent",
                    "intent": "raise_perigee",
                    "delta_perigee_m": 1500.0,
                },
            ),
            ManeuverRecord(
                name="raise-a-inc",
                trigger={"type": "ascending_node"},
                model={
                    "type": "intent",
                    "intent": "raise_semimajor_axis",
                    "delta_a_m": 2000.0,
                },
            ),
        ),
    )

    ctx = BuildContext.from_state_record(scenario.initial_state, universe=scenario.universe)
    initial_state = SpacecraftState(ctx.initial_orbit, float(scenario.initial_state.mass_kg or 450.0))
    compiled = CLIENT.compile_scenario_maneuvers(scenario, initial_state)
    assert len(compiled) == 2
    for item in compiled:
        norm = (item.dv_inertial_mps[0] ** 2 + item.dv_inertial_mps[1] ** 2 + item.dv_inertial_mps[2] ** 2) ** 0.5
        assert norm > 0.0


def test_export_scenario_with_maneuvers_and_plot(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None

    scenario_path = Path("examples/state_files/leo_intent_mission.yaml")
    scenario, propagator = _build_numerical_from_scenario(scenario_path)
    assert scenario.initial_state is not None

    epoch_spec = OutputEpochSpec(
        start_epoch=scenario.initial_state.epoch,
        end_epoch="2026-02-19T03:00:00Z",
        step_seconds=180.0,
    )
    out_series = tmp_path / "mission_series.yaml"
    saved_path, compiled = CLIENT.export_trajectory_from_scenario(
        propagator,
        scenario,
        epoch_spec,
        out_series,
        representation="keplerian",
        series_name="mission_profile",
    )

    assert saved_path.exists()
    assert len(compiled) == 2

    series = CLIENT.load_state_series(saved_path)
    initial = series.states[0]
    final = series.states[-1]
    assert _perigee_m(final) > _perigee_m(initial)
    initial_i = float((initial.elements or {})["i_deg"])
    final_i = float((final.elements or {})["i_deg"])
    assert final_i > initial_i

    out_png = tmp_path / "elements.png"
    saved_png = CLIENT.plot_orbital_elements(series, out_png)
    assert saved_png.exists()
    assert saved_png.stat().st_size > 0


def test_timeline_maintenance_semimajor_axis_floor(tmp_path: Path) -> None:
    scenario_path = Path("examples/state_files/leo_sma_maintenance_timeline.yaml")
    scenario, propagator = _build_numerical_from_scenario(scenario_path)
    assert scenario.initial_state is not None

    epoch_spec = OutputEpochSpec(
        start_epoch=scenario.initial_state.epoch,
        end_epoch="2026-02-19T02:30:00Z",
        step_seconds=180.0,
    )
    out_series = tmp_path / "maintenance_series.yaml"
    saved_path, compiled = CLIENT.export_trajectory_from_scenario(
        propagator,
        scenario,
        epoch_spec,
        out_series,
        representation="keplerian",
        series_name="maintenance_profile",
    )

    assert saved_path.exists()
    assert len(compiled) == 4
    by_name = {item.name: item for item in compiled}
    assert "maintain-a-1" in by_name
    dv1 = by_name["maintain-a-1"].dv_inertial_mps
    norm1 = (dv1[0] ** 2 + dv1[1] ** 2 + dv1[2] ** 2) ** 0.5
    assert norm1 > 0.0

    series = CLIENT.load_state_series(saved_path)
    floor_a = 6_876_000.0
    t_maint_1 = parse_epoch_utc(by_name["maintain-a-1"].epoch)
    post_maint_a = [_a_m(item) for item in series.states if parse_epoch_utc(item.epoch) >= t_maint_1]
    assert post_maint_a
    assert min(post_maint_a) >= floor_a
