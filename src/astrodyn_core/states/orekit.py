"""Conversion helpers between state-file records and Orekit objects."""

from __future__ import annotations

import math
from typing import Any, Mapping

from astrodyn_core.propagation.config import get_itrf_frame, get_mu
from astrodyn_core.states.models import OrbitStateRecord
from astrodyn_core.states.validation import parse_epoch_utc


def to_orekit_orbit(
    record: OrbitStateRecord,
    universe: Mapping[str, Any] | None = None,
):
    """Convert an OrbitStateRecord into an Orekit Orbit instance."""
    if not isinstance(record, OrbitStateRecord):
        raise TypeError("record must be an OrbitStateRecord.")

    try:
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit, PositionAngleType
        from org.orekit.utils import PVCoordinates
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    frame = resolve_frame(record.frame, universe=universe)
    date = to_orekit_date(record.epoch)
    mu = resolve_mu(record.mu_m3_s2, universe=universe)

    if record.representation == "cartesian":
        pos = Vector3D(*record.position_m)
        vel = Vector3D(*record.velocity_mps)
        pv = PVCoordinates(pos, vel)
        return CartesianOrbit(pv, frame, date, mu)

    if record.representation == "keplerian":
        el = record.elements or {}
        angle_type = getattr(PositionAngleType, el["anomaly_type"])
        return KeplerianOrbit(
            el["a_m"],
            el["e"],
            math.radians(el["i_deg"]),
            math.radians(el["argp_deg"]),
            math.radians(el["raan_deg"]),
            math.radians(el["anomaly_deg"]),
            angle_type,
            frame,
            date,
            mu,
        )

    if record.representation == "equinoctial":
        el = record.elements or {}
        angle_type = getattr(PositionAngleType, el["anomaly_type"])
        return EquinoctialOrbit(
            el["a_m"],
            el["ex"],
            el["ey"],
            el["hx"],
            el["hy"],
            math.radians(el["l_deg"]),
            angle_type,
            frame,
            date,
            mu,
        )

    raise ValueError(f"Unsupported representation '{record.representation}'.")


def to_orekit_date(epoch: str):
    """Convert an ISO-8601 epoch string into Orekit AbsoluteDate (UTC)."""
    parsed = parse_epoch_utc(epoch)
    try:
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    utc = TimeScalesFactory.getUTC()
    second = float(parsed.second) + parsed.microsecond / 1e6
    return AbsoluteDate(
        parsed.year,
        parsed.month,
        parsed.day,
        parsed.hour,
        parsed.minute,
        second,
        utc,
    )


def resolve_frame(frame_name: str, universe: Mapping[str, Any] | None = None):
    """Resolve a frame name from state files into an Orekit frame."""
    try:
        from org.orekit.frames import FramesFactory
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    normalized = frame_name.strip().upper()
    if normalized == "GCRF":
        return FramesFactory.getGCRF()
    if normalized == "EME2000":
        return FramesFactory.getEME2000()
    if normalized == "TEME":
        return FramesFactory.getTEME()

    if normalized in {"ITRF", "ITRF_2020", "ITRF_2014", "ITRF_2008"}:
        cfg = dict(universe or {})
        if normalized != "ITRF":
            cfg["itrf_version"] = normalized
        return get_itrf_frame(cfg if cfg else None)

    raise ValueError(f"Unsupported frame '{frame_name}'.")


def resolve_mu(mu_m3_s2: float | str, universe: Mapping[str, Any] | None = None) -> float:
    """Resolve a numeric gravitational parameter in m^3/s^2."""
    if isinstance(mu_m3_s2, (int, float)):
        return float(mu_m3_s2)

    if isinstance(mu_m3_s2, str):
        cfg = dict(universe or {})
        cfg["gravitational_parameter"] = mu_m3_s2.strip().upper()
        return get_mu(cfg)

    raise TypeError("mu_m3_s2 must be a float or predefined model string.")
