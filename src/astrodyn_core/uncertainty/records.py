"""Record conversion helpers for uncertainty propagation outputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from astrodyn_core.states.models import OrbitStateRecord
from astrodyn_core.states.orekit_dates import from_orekit_date


def numpy_to_nested_tuple(arr: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Convert a NumPy matrix into nested tuples.

    Args:
        arr: Input matrix.

    Returns:
        Tuple-of-tuples representation with float values.
    """
    return tuple(tuple(float(v) for v in row) for row in arr)


def state_to_orbit_record(
    state: Any,
    *,
    frame: str,
    orbit_type: str,
    mu_m3_s2: float | str,
    default_mass_kg: float,
    output_frame: Any | None = None,
) -> OrbitStateRecord:
    """Convert an Orekit ``SpacecraftState`` into an ``OrbitStateRecord``.

    Args:
        state: Orekit ``SpacecraftState``.
        frame: Output frame label stored in the record.
        orbit_type: Desired record representation (``CARTESIAN``,
            ``KEPLERIAN``, or ``EQUINOCTIAL``).
        mu_m3_s2: Gravitational parameter attached to the record.
        default_mass_kg: Fallback mass when ``state`` has no mass accessor.
        output_frame: Optional Orekit frame used to first transform the state.

    Returns:
        Orbit state record in the requested representation.
    """
    from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit

    mass = float(state.getMass()) if hasattr(state, "getMass") else float(default_mass_kg)
    epoch = from_orekit_date(state.getDate())
    orbit = state.getOrbit()
    if output_frame is not None and orbit.getFrame() != output_frame:
        pv_out = state.getPVCoordinates(output_frame)
        orbit = CartesianOrbit(pv_out, output_frame, state.getDate(), float(orbit.getMu()))

    ot = orbit_type.upper()
    if ot == "CARTESIAN":
        pv = state.getPVCoordinates(orbit.getFrame())
        pos = pv.getPosition()
        vel = pv.getVelocity()
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame,
            representation="cartesian",
            position_m=(float(pos.getX()), float(pos.getY()), float(pos.getZ())),
            velocity_mps=(float(vel.getX()), float(vel.getY()), float(vel.getZ())),
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    if ot == "KEPLERIAN":
        kep = KeplerianOrbit(orbit)
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame,
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

    equi = EquinoctialOrbit(orbit)
    return OrbitStateRecord(
        epoch=epoch,
        frame=frame,
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
