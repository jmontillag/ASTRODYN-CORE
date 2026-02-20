"""Shared helpers for example entry-point scripts."""

from __future__ import annotations

from pathlib import Path

_OREKIT_INITIALIZED = False


def init_orekit() -> None:
    """Initialize JVM + Orekit data once for the current process."""
    global _OREKIT_INITIALIZED
    if _OREKIT_INITIALIZED:
        return

    import orekit

    orekit.initVM()
    from orekit.pyhelpers import setup_orekit_curdir

    setup_orekit_curdir()
    _OREKIT_INITIALIZED = True


def build_factory():
    """Return a PropagatorFactory with default Orekit providers registered."""
    from astrodyn_core import PropagatorFactory, ProviderRegistry, register_default_orekit_providers

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    return PropagatorFactory(registry=registry)


def make_generated_dir() -> Path:
    """Return the examples generated-output directory, creating it if needed."""
    out = Path(__file__).resolve().parent / "generated"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_leo_orbit(
    *,
    year: int = 2026,
    month: int = 2,
    day: int = 19,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
):
    """Create a representative LEO Keplerian orbit and return (orbit, epoch, frame)."""
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import Constants

    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(year, month, day, hour, minute, second, utc)
    frame = FramesFactory.getGCRF()

    orbit = KeplerianOrbit(
        6_878_137.0,
        0.0012,
        0.9005898940290741,  # 51.6 deg
        0.7853981633974483,  # 45 deg
        2.0943951023931953,  # 120 deg
        0.0,
        PositionAngleType.MEAN,
        frame,
        epoch,
        Constants.WGS84_EARTH_MU,
    )
    return orbit, epoch, frame
