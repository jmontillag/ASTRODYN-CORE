import pytest

from astrodyn_core.states import OrbitStateRecord, StateFileClient

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

CLIENT = StateFileClient()


def test_cartesian_state_conversion_round_trip() -> None:
    record = OrbitStateRecord(
        epoch="2026-02-19T00:00:00Z",
        frame="GCRF",
        representation="cartesian",
        position_m=(7000000.0, -1200.0, 500.0),
        velocity_mps=(10.0, 7500.0, -2.0),
        mu_m3_s2="WGS84",
    )

    orbit = CLIENT.to_orekit_orbit(record)
    pv = orbit.getPVCoordinates()
    pos = pv.getPosition()
    vel = pv.getVelocity()

    assert pos.getX() == pytest.approx(record.position_m[0], abs=1e-6)
    assert pos.getY() == pytest.approx(record.position_m[1], abs=1e-6)
    assert pos.getZ() == pytest.approx(record.position_m[2], abs=1e-6)
    assert vel.getX() == pytest.approx(record.velocity_mps[0], abs=1e-9)
    assert vel.getY() == pytest.approx(record.velocity_mps[1], abs=1e-9)
    assert vel.getZ() == pytest.approx(record.velocity_mps[2], abs=1e-9)
