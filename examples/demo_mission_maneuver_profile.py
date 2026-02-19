#!/usr/bin/env python
"""Run a mission scenario with intent maneuvers and plot element evolution.

Run from repo root:
    python examples/demo_mission_maneuver_profile.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from astrodyn_core import (  # noqa: E402
    BuildContext,
    IntegratorSpec,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    StateFileClient,
    register_default_orekit_providers,
)
from astrodyn_core.states.validation import parse_epoch_utc  # noqa: E402


def _perigee_from_keplerian_state(state) -> float:
    elements = state.elements or {}
    return float(elements["a_m"]) * (1.0 - float(elements["e"]))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    scenario_path = base_dir / "state_files" / "leo_intent_mission.yaml"
    output_series_path = base_dir / "state_files" / "mission_profile_series.yaml"
    output_plot_path = base_dir / "state_files" / "mission_profile_elements.png"

    client = StateFileClient()
    scenario = client.load_state_file(scenario_path)
    if scenario.initial_state is None:
        raise RuntimeError("Scenario must define initial_state.")

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

    start_dt = parse_epoch_utc(scenario.initial_state.epoch)
    end_epoch = (start_dt + timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    epoch_spec = OutputEpochSpec(
        start_epoch=scenario.initial_state.epoch,
        end_epoch=end_epoch,
        step_seconds=120.0,
    )

    saved_path, compiled = client.export_trajectory_from_scenario(
        propagator,
        scenario,
        epoch_spec,
        output_series_path,
        series_name="mission_profile",
        representation="keplerian",
        frame="GCRF",
        dense_yaml=True,
    )
    series = client.load_state_series(saved_path)
    client.plot_orbital_elements(series, output_plot_path, title="Mission Profile: Orbital Elements")

    initial_state = series.states[0]
    final_state = series.states[-1]
    initial_rp = _perigee_from_keplerian_state(initial_state)
    final_rp = _perigee_from_keplerian_state(final_state)
    initial_i = float((initial_state.elements or {})["i_deg"])
    final_i = float((final_state.elements or {})["i_deg"])

    print("Mission profile with maneuvers complete")
    print(f"Scenario: {scenario_path.name}")
    print(f"Saved trajectory: {saved_path.name}")
    print(f"Saved element plot: {output_plot_path.name}")
    print(f"Compiled maneuvers: {len(compiled)}")
    for item in compiled:
        dv_norm = (item.dv_inertial_mps[0] ** 2 + item.dv_inertial_mps[1] ** 2 + item.dv_inertial_mps[2] ** 2) ** 0.5
        print(f"  - {item.name}: epoch={item.epoch}, trigger={item.trigger_type}, |dv|={dv_norm:.3f} m/s")
    print(f"Perigee change: {initial_rp:.2f} m -> {final_rp:.2f} m")
    print(f"Inclination change: {initial_i:.4f} deg -> {final_i:.4f} deg")


if __name__ == "__main__":
    main()
