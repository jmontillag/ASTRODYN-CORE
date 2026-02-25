"""State-record and Orekit orbit/state conversion helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

from astrodyn_core.states.models import OrbitStateRecord
from astrodyn_core.states.orekit_dates import to_orekit_date
from astrodyn_core.states.orekit_resolvers import resolve_frame, resolve_mu


def to_orekit_orbit(
    record: OrbitStateRecord,
    universe: Mapping[str, Any] | None = None,
):
    """Convert an ``OrbitStateRecord`` into an Orekit orbit instance.

    Args:
        record: Serializable orbit-state record.
        universe: Optional universe config used by frame and ``mu`` resolvers.

    Returns:
        Orekit orbit object matching the record representation.

    Raises:
        TypeError: If ``record`` is not an ``OrbitStateRecord``.
        RuntimeError: If Orekit classes are unavailable.
        ValueError: If the state representation is unsupported.
    """
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


def state_to_record(
    state: Any,
    *,
    epoch: str,
    representation: str,
    frame_name: str,
    output_frame: Any,
    mu_m3_s2: float | str,
    default_mass_kg: float,
) -> OrbitStateRecord:
    """Convert an Orekit state to a serializable ``OrbitStateRecord``.

    Args:
        state: Orekit ``SpacecraftState``.
        epoch: Output epoch string for the serialized record.
        representation: Output representation (cartesian/keplerian/equinoctial).
        frame_name: Output frame label stored in the record.
        output_frame: Orekit frame used for extracting coordinates.
        mu_m3_s2: Gravitational parameter value/key stored in the record.
        default_mass_kg: Fallback mass when the state has no mass accessor.

    Returns:
        Serializable orbit-state record in the requested representation.

    Raises:
        RuntimeError: If Orekit classes are unavailable.
    """
    try:
        from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    mass = float(state.getMass()) if hasattr(state, "getMass") else float(default_mass_kg)
    orbit = state.getOrbit()
    mu = orbit.getMu()

    if representation == "cartesian":
        pv = state.getPVCoordinates(output_frame)
        pos = pv.getPosition()
        vel = pv.getVelocity()
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame_name,
            representation="cartesian",
            position_m=(pos.getX(), pos.getY(), pos.getZ()),
            velocity_mps=(vel.getX(), vel.getY(), vel.getZ()),
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    orbit_in_frame = orbit
    if orbit.getFrame() != output_frame:
        pv = state.getPVCoordinates(output_frame)
        orbit_in_frame = CartesianOrbit(pv, output_frame, state.getDate(), mu)

    if representation == "keplerian":
        kep = KeplerianOrbit(orbit_in_frame)
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame_name,
            representation="keplerian",
            elements={
                "a_m": float(kep.getA()),
                "e": float(kep.getE()),
                "i_deg": math.degrees(float(kep.getI())),
                "argp_deg": math.degrees(float(kep.getPerigeeArgument())),
                "raan_deg": math.degrees(float(kep.getRightAscensionOfAscendingNode())),
                "anomaly_deg": math.degrees(float(kep.getMeanAnomaly())),
                "anomaly_type": "MEAN",
            },
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    equi = EquinoctialOrbit(orbit_in_frame)
    return OrbitStateRecord(
        epoch=epoch,
        frame=frame_name,
        representation="equinoctial",
        elements={
            "a_m": float(equi.getA()),
            "ex": float(equi.getEquinoctialEx()),
            "ey": float(equi.getEquinoctialEy()),
            "hx": float(equi.getHx()),
            "hy": float(equi.getHy()),
            "l_deg": math.degrees(float(equi.getLM())),
            "anomaly_type": "MEAN",
        },
        mu_m3_s2=mu_m3_s2,
        mass_kg=mass,
    )
