from __future__ import annotations

import math
from pathlib import Path

import pytest

from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    register_default_orekit_providers,
    StateFileClient,
)

CLIENT = StateFileClient()

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import KeplerianOrbit, PositionAngleType  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402
from org.orekit.utils import Constants  # noqa: E402


def _build_numerical_propagator():
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2026, 2, 19, 0, 0, 0.0, utc)
    gcrf = FramesFactory.getGCRF()
    orbit = KeplerianOrbit(
        6_878_137.0,
        0.0012,
        math.radians(51.6),
        math.radians(45.0),
        math.radians(120.0),
        0.0,
        PositionAngleType.MEAN,
        gcrf,
        epoch,
        Constants.WGS84_EARTH_MU,
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        mass_kg=450.0,
        integrator=IntegratorSpec(
            kind="dp54",
            min_step=1.0e-6,
            max_step=300.0,
            position_tolerance=1.0e-3,
        ),
    )
    ctx = BuildContext(initial_orbit=orbit)
    builder = factory.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    return propagator, gcrf


def _build_precomputed_ephemeris(end_epoch: str = "2026-02-19T00:20:00Z"):
    propagator, gcrf = _build_numerical_propagator()
    generator = propagator.getEphemerisGenerator()
    propagator.propagate(CLIENT.to_orekit_date(end_epoch))
    return generator.getGeneratedEphemeris(), gcrf


def test_export_wrapper_yaml_and_off_grid_compare(tmp_path: Path) -> None:
    exporter_propagator, _ = _build_numerical_propagator()
    baseline_propagator, gcrf = _build_numerical_propagator()

    epoch_spec = OutputEpochSpec(
        start_epoch="2026-02-19T00:00:00Z",
        end_epoch="2026-02-19T00:20:00Z",
        step_seconds=120.0,
    )
    out_file = tmp_path / "traj.yaml"
    CLIENT.export_trajectory_from_propagator(exporter_propagator, epoch_spec, out_file, series_name="traj")

    scenario = CLIENT.load_state_file(out_file)
    assert len(scenario.state_series) == 1
    series = scenario.state_series[0]
    assert len(series.states) == 11

    ephemeris = CLIENT.scenario_to_ephemeris(scenario)
    for offset_s in (60, 300, 780):
        date = CLIENT.to_orekit_date(f"2026-02-19T00:{offset_s // 60:02d}:00Z")
        truth = baseline_propagator.propagate(date).getPVCoordinates(gcrf).getPosition()
        interp = ephemeris.propagate(date).getPVCoordinates(gcrf).getPosition()
        error_m = truth.subtract(interp).getNorm()
        assert error_m < 100.0


def test_export_wrapper_hdf5_explicit_epochs(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    assert h5py is not None

    propagator, _ = _build_numerical_propagator()
    epoch_spec = OutputEpochSpec(
        explicit_epochs=(
            "2026-02-19T00:00:00Z",
            "2026-02-19T00:03:00Z",
            "2026-02-19T00:09:00Z",
        )
    )
    out_file = tmp_path / "traj.h5"
    CLIENT.export_trajectory_from_propagator(
        propagator,
        epoch_spec,
        out_file,
        representation="keplerian",
        series_name="traj_h5",
    )

    series = CLIENT.load_state_series(out_file)
    assert series.name == "traj_h5"
    assert len(series.states) == 3
    assert series.states[0].representation == "keplerian"


def test_export_wrapper_accepts_precomputed_ephemeris(tmp_path: Path) -> None:
    ephemeris, gcrf = _build_precomputed_ephemeris()

    epoch_spec = OutputEpochSpec(
        start_epoch="2026-02-19T00:00:00Z",
        end_epoch="2026-02-19T00:20:00Z",
        step_seconds=120.0,
    )
    out_file = tmp_path / "traj_ephem.yaml"
    CLIENT.export_trajectory_from_propagator(ephemeris, epoch_spec, out_file, series_name="traj_ephem")

    scenario = CLIENT.load_state_file(out_file)
    assert len(scenario.state_series) == 1
    assert len(scenario.state_series[0].states) == 11

    target_date = CLIENT.to_orekit_date("2026-02-19T00:07:00Z")
    truth = ephemeris.propagate(target_date).getPVCoordinates(gcrf).getPosition()
    interp = CLIENT.scenario_to_ephemeris(scenario).propagate(target_date).getPVCoordinates(gcrf).getPosition()
    assert truth.subtract(interp).getNorm() < 100.0


def test_export_wrapper_precomputed_ephemeris_out_of_bounds_rejected(tmp_path: Path) -> None:
    ephemeris, _ = _build_precomputed_ephemeris(end_epoch="2026-02-19T00:10:00Z")
    epoch_spec = OutputEpochSpec(
        start_epoch="2026-02-19T00:00:00Z",
        end_epoch="2026-02-19T00:20:00Z",
        step_seconds=120.0,
    )

    with pytest.raises(ValueError, match="outside the available range"):
        CLIENT.export_trajectory_from_propagator(ephemeris, epoch_spec, tmp_path / "oob.yaml")
