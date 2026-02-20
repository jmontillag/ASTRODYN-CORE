#!/usr/bin/env python
"""Scenario and mission workflows for ASTRODYN-CORE.

Covers four scenario-centric workflows:
1. `io`: state-file save/load + interpolation accuracy checks
2. `inspect`: read mission timeline metadata and enrich scenario outputs
3. `intent`: compile and replay intent maneuvers from a scenario file
4. `detector`: execute maneuvers via Orekit event detectors (closed-loop)

Run from project root:
    python examples/scenario_missions.py --mode all
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _common import build_factory, init_orekit, make_generated_dir, make_leo_orbit


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _parse_epoch_utc(epoch: str) -> datetime:
    normalized = epoch.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _active_attitude_mode_at(scenario, epoch_utc: datetime) -> str:
    selected_mode = "unknown"
    selected_start: datetime | None = None

    for entry in scenario.attitude_timeline:
        start_raw = entry.params.get("start_epoch")
        if not isinstance(start_raw, str):
            continue
        start_dt = _parse_epoch_utc(start_raw)
        if start_dt <= epoch_utc and (selected_start is None or start_dt > selected_start):
            selected_mode = entry.mode
            selected_start = start_dt
    return selected_mode


def _build_numerical_propagator(initial_state, *, universe=None, integrator_kind: str = "dp853"):
    from astrodyn_core import BuildContext, IntegratorSpec, PropagatorKind, PropagatorSpec

    factory = build_factory()
    ctx = BuildContext.from_state_record(initial_state, universe=universe)
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        mass_kg=float(initial_state.mass_kg or 450.0),
        integrator=IntegratorSpec(
            kind=integrator_kind,
            min_step=1.0e-6,
            max_step=300.0,
            position_tolerance=1.0e-3,
        ),
    )
    builder = factory.build_builder(spec, ctx)
    return builder.buildPropagator(builder.getSelectedNormalizedParameters())


def run_io() -> None:
    _header("Scenario Missions · I/O Workflow")
    init_orekit()

    from astrodyn_core import (
        BuildContext,
        OrbitStateRecord,
        OutputEpochSpec,
        PropagatorKind,
        PropagatorSpec,
        StateFileClient,
    )

    out_dir = make_generated_dir()
    client = StateFileClient()

    orbit, _epoch, gcrf = make_leo_orbit()
    factory = build_factory()
    builder = factory.build_builder(
        PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0),
        BuildContext(initial_orbit=orbit),
    )
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    initial_epoch = "2026-02-19T00:00:00Z"
    state0 = propagator.propagate(client.to_orekit_date(initial_epoch))
    pv0 = state0.getPVCoordinates(gcrf)
    initial_record = OrbitStateRecord(
        epoch=initial_epoch,
        frame="GCRF",
        representation="cartesian",
        position_m=(pv0.getPosition().getX(), pv0.getPosition().getY(), pv0.getPosition().getZ()),
        velocity_mps=(pv0.getVelocity().getX(), pv0.getVelocity().getY(), pv0.getVelocity().getZ()),
        mu_m3_s2="WGS84",
        mass_kg=state0.getMass(),
    )

    initial_file = out_dir / "workflow_initial_state.yaml"
    yaml_series_file = out_dir / "workflow_cartesian_series.yaml"
    h5_series_file = out_dir / "workflow_cartesian_series.h5"

    client.save_initial_state(initial_file, initial_record)
    loaded_initial = client.load_state_file(initial_file).initial_state
    if loaded_initial is None:
        raise RuntimeError("Failed to load initial state from generated file.")

    start_dt = _parse_epoch_utc(loaded_initial.epoch)
    end_epoch = (start_dt + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    epoch_spec = OutputEpochSpec(
        start_epoch=loaded_initial.epoch,
        end_epoch=end_epoch,
        step_seconds=120.0,
    )

    client.export_trajectory_from_propagator(
        propagator,
        epoch_spec,
        yaml_series_file,
        series_name="workflow_cartesian_arc",
        representation="cartesian",
        frame="GCRF",
        dense_yaml=True,
    )
    client.export_trajectory_from_propagator(
        propagator,
        epoch_spec,
        h5_series_file,
        series_name="workflow_cartesian_arc",
        representation="cartesian",
        frame="GCRF",
    )

    yaml_scenario = client.load_state_file(yaml_series_file)
    yaml_ephemeris = client.scenario_to_ephemeris(yaml_scenario)
    h5_series = client.load_state_series(h5_series_file)
    h5_ephemeris = client.state_series_to_ephemeris(h5_series)

    sample_step_s = 120
    off_grid_offsets = [60, 300, 780, 1500, 2460]
    max_yaml_err_m = 0.0
    max_h5_err_m = 0.0

    for offset_s in off_grid_offsets:
        if offset_s % sample_step_s == 0:
            continue
        epoch_str = (start_dt + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")
        date = client.to_orekit_date(epoch_str)
        truth = propagator.propagate(date).getPVCoordinates(gcrf).getPosition()
        from_yaml = yaml_ephemeris.propagate(date).getPVCoordinates(gcrf).getPosition()
        from_h5 = h5_ephemeris.propagate(date).getPVCoordinates(gcrf).getPosition()
        yaml_err = truth.subtract(from_yaml).getNorm()
        h5_err = truth.subtract(from_h5).getNorm()
        max_yaml_err_m = max(max_yaml_err_m, yaml_err)
        max_h5_err_m = max(max_h5_err_m, h5_err)

    print(f"Initial state: {initial_file}")
    print(f"YAML series: {yaml_series_file}")
    print(f"HDF5 series: {h5_series_file}")
    print(f"Max YAML interpolation error: {max_yaml_err_m:.6f} m")
    print(f"Max HDF5 interpolation error: {max_h5_err_m:.6f} m")


def run_inspect() -> None:
    _header("Scenario Missions · Inspect + Enrich")
    init_orekit()

    from astrodyn_core import (
        BuildContext,
        PropagatorKind,
        PropagatorSpec,
        ScenarioStateFile,
        StateFileClient,
    )

    base_dir = Path(__file__).resolve().parent
    scenario_path = base_dir / "state_files" / "leo_mission_timeline.yaml"
    out_path = make_generated_dir() / "scenario_enriched.yaml"

    client = StateFileClient()
    scenario = client.load_state_file(scenario_path)
    if scenario.initial_state is None:
        raise RuntimeError("Scenario requires initial_state.")

    print(f"Mission: {scenario.metadata.get('mission', 'unknown')}")
    print(f"Maneuvers: {len(scenario.maneuvers)}")
    print(f"Attitude entries: {len(scenario.attitude_timeline)}")

    factory = build_factory()
    ctx = BuildContext.from_state_record(
        scenario.initial_state,
        universe=scenario.universe,
        metadata={"source_scenario": scenario_path.name},
    )
    builder = factory.build_builder(
        PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=float(scenario.initial_state.mass_kg or 450.0)),
        ctx,
    )
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    previewed = 0
    for maneuver in scenario.maneuvers:
        trigger_type = str(maneuver.trigger.get("type", "")).strip().lower()
        epoch_raw = maneuver.trigger.get("epoch")
        if trigger_type != "epoch" or not isinstance(epoch_raw, str):
            continue

        epoch_dt = _parse_epoch_utc(epoch_raw)
        state = propagator.propagate(client.to_orekit_date(epoch_raw))
        r_km = state.getPVCoordinates().getPosition().getNorm() / 1000.0
        mode = _active_attitude_mode_at(scenario, epoch_dt)
        print(f"{maneuver.name}: epoch={epoch_raw}, |r|={r_km:.3f} km, attitude={mode}")
        previewed += 1

    enriched_metadata = dict(scenario.metadata)
    enriched_metadata["derived_preview_maneuver_count"] = previewed
    enriched_metadata["derived_preview_frame"] = scenario.initial_state.frame

    enriched = ScenarioStateFile(
        schema_version=scenario.schema_version,
        universe=scenario.universe,
        spacecraft=scenario.spacecraft,
        initial_state=scenario.initial_state,
        timeline=scenario.timeline,
        state_series=scenario.state_series,
        maneuvers=scenario.maneuvers,
        attitude_timeline=scenario.attitude_timeline,
        metadata=enriched_metadata,
    )
    client.save_state_file(out_path, enriched)
    print(f"Saved enriched scenario: {out_path}")


def run_intent() -> None:
    _header("Scenario Missions · Intent Maneuvers")
    init_orekit()

    from astrodyn_core import OutputEpochSpec, StateFileClient
    from astrodyn_core.states.validation import parse_epoch_utc

    base_dir = Path(__file__).resolve().parent
    scenario_path = base_dir / "state_files" / "leo_intent_mission.yaml"
    out_dir = make_generated_dir()
    out_series = out_dir / "mission_profile_series.yaml"
    out_plot = out_dir / "mission_profile_elements.png"

    client = StateFileClient()
    scenario = client.load_state_file(scenario_path)
    if scenario.initial_state is None:
        raise RuntimeError("Scenario must define initial_state.")

    propagator = _build_numerical_propagator(
        scenario.initial_state,
        universe=scenario.universe,
        integrator_kind="dp54",
    )

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
        out_series,
        series_name="mission_profile",
        representation="keplerian",
        frame="GCRF",
        dense_yaml=True,
    )
    series = client.load_state_series(saved_path)
    client.plot_orbital_elements(series, out_plot, title="Mission Profile: Orbital Elements")

    print(f"Saved trajectory: {saved_path}")
    print(f"Saved plot: {out_plot}")
    print(f"Compiled maneuvers: {len(compiled)}")
    for item in compiled:
        dv_norm = (
            item.dv_inertial_mps[0] ** 2
            + item.dv_inertial_mps[1] ** 2
            + item.dv_inertial_mps[2] ** 2
        ) ** 0.5
        print(
            f"  - {item.name}: epoch={item.epoch}, trigger={item.trigger_type}, "
            f"|dv|={dv_norm:.3f} m/s"
        )


def run_detector() -> None:
    _header("Scenario Missions · Detector-Driven Execution")
    init_orekit()

    from astrodyn_core import OutputEpochSpec, StateFileClient

    base_dir = Path(__file__).resolve().parent
    scenario_path = base_dir / "state_files" / "leo_detector_mission.yaml"
    out_dir = make_generated_dir()
    out_states = out_dir / "detector_mission_trajectory.yaml"
    out_plot = out_dir / "detector_mission_elements.png"

    client = StateFileClient()
    scenario = client.load_state_file(scenario_path)
    if scenario.initial_state is None:
        raise RuntimeError("Scenario must define initial_state.")

    propagator = _build_numerical_propagator(
        scenario.initial_state,
        universe=scenario.universe,
        integrator_kind="dp853",
    )

    epoch_spec = OutputEpochSpec(
        start_epoch="2026-02-19T00:00:00Z",
        end_epoch="2026-02-19T06:00:00Z",
        step_seconds=300.0,
    )

    state_series, report = client.run_scenario_detector_mode(
        propagator,
        scenario,
        epoch_spec,
        representation="keplerian",
        output_path=out_states,
    )

    print(f"Events fired: {len(report.events)}")
    print(f"Applied: {len(report.applied_events())}")
    print(f"Skipped: {len(report.skipped_events())}")
    print(f"Total Δv: {report.total_dv_mps:.4f} m/s")

    if report.events:
        for event in report.events:
            status = "APPLIED" if event.applied else f"SKIPPED ({event.guard_skip_reason})"
            print(f"  - [{event.epoch}] {event.maneuver_name}: {status}")

    if state_series.states:
        client.plot_orbital_elements(state_series, out_plot)
        print(f"Saved plot: {out_plot}")
    print(f"Saved trajectory: {out_states}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario + mission example workflows")
    parser.add_argument(
        "--mode",
        choices=("all", "io", "inspect", "intent", "detector"),
        default="all",
        help="Choose one workflow or run all (default).",
    )
    args = parser.parse_args()

    steps = {
        "io": run_io,
        "inspect": run_inspect,
        "intent": run_intent,
        "detector": run_detector,
    }

    if args.mode == "all":
        for key in ("io", "inspect", "intent", "detector"):
            steps[key]()
    else:
        steps[args.mode]()


if __name__ == "__main__":
    main()
