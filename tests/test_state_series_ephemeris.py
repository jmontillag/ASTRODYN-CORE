from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from astrodyn_core import (
    BuildContext,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    StateFileClient,
    register_default_orekit_providers,
)
from astrodyn_core.states import (
    OrbitStateRecord,
    StateSeries,
)

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import KeplerianOrbit, PositionAngleType  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402
from org.orekit.utils import Constants  # noqa: E402

CLIENT = StateFileClient()


def test_epoch_string_roundtrip_with_orekit_helpers() -> None:
    epoch = "2026-02-19T00:00:00Z"
    orekit_date = CLIENT.to_orekit_date(epoch)
    converted = CLIENT.from_orekit_date(orekit_date)
    assert converted == epoch


def test_generated_series_file_converts_to_ephemeris_and_interpolates(tmp_path: Path) -> None:
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2026, 2, 19, 0, 0, 0.0, utc)
    gcrf = FramesFactory.getGCRF()

    initial_orbit = KeplerianOrbit(
        6_878_137.0,
        0.0012,
        0.9005898940290741,  # rad (51.6 deg)
        0.7853981633974483,  # rad (45 deg)
        2.0943951023931953,  # rad (120 deg)
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
    baseline = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    start_dt = datetime(2026, 2, 19, 0, 0, 0, tzinfo=timezone.utc)
    sample_step_s = 120
    duration_s = 3600
    n_samples = duration_s // sample_step_s + 1

    records: list[OrbitStateRecord] = []
    for idx in range(n_samples):
        offset_s = idx * sample_step_s
        epoch_str = (start_dt + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")
        date = CLIENT.to_orekit_date(epoch_str)
        state = baseline.propagate(date)
        pv = state.getPVCoordinates(gcrf)
        pos = pv.getPosition()
        vel = pv.getVelocity()

        records.append(
            OrbitStateRecord(
                epoch=epoch_str,
                frame="GCRF",
                representation="cartesian",
                position_m=(pos.getX(), pos.getY(), pos.getZ()),
                velocity_mps=(vel.getX(), vel.getY(), vel.getZ()),
                mu_m3_s2="WGS84",
                mass_kg=state.getMass(),
            )
        )

    series = StateSeries(
        name="generated_cartesian_arc",
        states=tuple(records),
        interpolation={"method": "orekit_ephemeris", "samples": 8},
    )

    out_file = tmp_path / "generated_cartesian_series.yaml"
    CLIENT.save_state_series(out_file, series, dense_yaml=True)

    scenario = CLIENT.load_state_file(out_file)
    loaded_series = scenario.state_series[0]
    ephemeris = CLIENT.state_series_to_ephemeris(loaded_series)

    # Off-grid times: not on the 2-minute sampling nodes
    off_grid_offsets = (60, 300, 780, 1500, 2460)
    for offset_s in off_grid_offsets:
        assert offset_s % sample_step_s != 0
        target = CLIENT.to_orekit_date(
            (start_dt + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")
        )

        truth_state = baseline.propagate(target)
        interp_state = ephemeris.propagate(target)

        truth_pos = truth_state.getPVCoordinates(gcrf).getPosition()
        interp_pos = interp_state.getPVCoordinates(gcrf).getPosition()
        error_m = truth_pos.subtract(interp_pos).getNorm()

        assert error_m < 300.0
