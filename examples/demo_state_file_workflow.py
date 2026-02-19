#!/usr/bin/env python
"""Unified file workflow demo for ASTRODYN-CORE state I/O.

Covers:
1. Create/save/load a single initial state (YAML)
2. Propagate and sample a state series every 2 minutes
3. Save/load compact dense YAML state series
4. Save/load HDF5 state series (compressed)
5. Convert loaded series to Orekit ephemeris interpolators
6. Compare interpolated positions against baseline at off-grid epochs
7. Load maneuver/attitude timeline file and print parsed entries

Run from repo root:
    python examples/demo_state_file_workflow.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import KeplerianOrbit, PositionAngleType  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402
from org.orekit.utils import Constants  # noqa: E402

from astrodyn_core import (  # noqa: E402
    BuildContext,
    OrbitStateRecord,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    StateFileClient,
    register_default_orekit_providers,
)
from astrodyn_core.states.validation import parse_epoch_utc  # noqa: E402


def _make_baseline_propagator():
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2026, 2, 19, 0, 0, 0.0, utc)
    gcrf = FramesFactory.getGCRF()
    initial_orbit = KeplerianOrbit(
        6_878_137.0,
        0.0012,
        0.9005898940290741,  # 51.6 deg
        0.7853981633974483,  # 45 deg
        2.0943951023931953,  # 120 deg
        0.0,
        PositionAngleType.MEAN,
        gcrf,
        epoch,
        Constants.WGS84_EARTH_MU,
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    spec = PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0)
    ctx = BuildContext(initial_orbit=initial_orbit)
    builder = factory.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    return propagator, gcrf


def main() -> None:
    state_dir = Path(__file__).resolve().parent / "state_files"
    state_dir.mkdir(parents=True, exist_ok=True)
    client = StateFileClient()

    propagator, gcrf = _make_baseline_propagator()

    # 1) Single-state file save/load
    initial_epoch = "2026-02-19T00:00:00Z"
    initial_state = propagator.propagate(client.to_orekit_date(initial_epoch))
    initial_pv = initial_state.getPVCoordinates(gcrf)
    initial_record = OrbitStateRecord(
        epoch=initial_epoch,
        frame="GCRF",
        representation="cartesian",
        position_m=(
            initial_pv.getPosition().getX(),
            initial_pv.getPosition().getY(),
            initial_pv.getPosition().getZ(),
        ),
        velocity_mps=(
            initial_pv.getVelocity().getX(),
            initial_pv.getVelocity().getY(),
            initial_pv.getVelocity().getZ(),
        ),
        mu_m3_s2="WGS84",
        mass_kg=initial_state.getMass(),
    )

    initial_file = state_dir / "workflow_initial_state.yaml"
    client.save_initial_state(initial_file, initial_record)
    loaded_initial = client.load_state_file(initial_file).initial_state
    if loaded_initial is None:
        raise RuntimeError("Failed to load initial state from workflow file.")

    # 2) Define output instants for trajectory file export
    start_dt = parse_epoch_utc(loaded_initial.epoch)
    sample_step_s = 120
    duration_s = 3600
    end_epoch = (start_dt + timedelta(seconds=duration_s)).isoformat().replace("+00:00", "Z")
    epoch_spec = OutputEpochSpec(
        start_epoch=loaded_initial.epoch,
        end_epoch=end_epoch,
        step_seconds=sample_step_s,
    )

    # 3) Export/load dense YAML series using wrapper
    yaml_series_file = state_dir / "workflow_cartesian_series.yaml"
    client.export_trajectory_from_propagator(
        propagator,
        epoch_spec,
        yaml_series_file,
        series_name="workflow_cartesian_arc",
        representation="cartesian",
        frame="GCRF",
        interpolation_samples=8,
        dense_yaml=True,
    )
    yaml_scenario = client.load_state_file(yaml_series_file)
    yaml_ephemeris = client.scenario_to_ephemeris(yaml_scenario)
    yaml_series = yaml_scenario.state_series[0]

    # 4) Export/load HDF5 series using wrapper
    h5_series_file = state_dir / "workflow_cartesian_series.h5"
    client.export_trajectory_from_propagator(
        propagator,
        epoch_spec,
        h5_series_file,
        series_name="workflow_cartesian_arc",
        representation="cartesian",
        frame="GCRF",
        interpolation_samples=8,
    )
    h5_series = client.load_state_series(h5_series_file)
    h5_ephemeris = client.state_series_to_ephemeris(h5_series)

    # 5) Compare at off-grid epochs
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

    # 6) Load timeline example
    timeline_file = state_dir / "leo_mission_timeline.yaml"
    timeline = client.load_state_file(timeline_file)

    print("Unified file workflow complete")
    print(f"Initial state file: {initial_file.name}")
    print(f"Dense YAML series:  {yaml_series_file.name}")
    print(f"HDF5 series:        {h5_series_file.name}")
    print(f"Series samples:     {len(yaml_series.states)} (step={sample_step_s}s)")
    print(f"Max YAML interp error (off-grid): {max_yaml_err_m:.6f} m")
    print(f"Max HDF5 interp error (off-grid): {max_h5_err_m:.6f} m")
    print(
        f"Timeline parsed: maneuvers={len(timeline.maneuvers)}, "
        f"attitude entries={len(timeline.attitude_timeline)}"
    )


if __name__ == "__main__":
    main()
