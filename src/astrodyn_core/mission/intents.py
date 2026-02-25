"""Intent and impulsive maneuver delta-v resolution logic."""

from __future__ import annotations

import math
from typing import Any, Mapping

from astrodyn_core.mission.kinematics import (
    local_to_inertial_delta_v,
    rotate_vector_about_axis,
    to_vector_tuple,
    unit,
)


def resolve_delta_v_vector(model: Mapping[str, Any], state: Any, trigger_type: str):
    """Resolve a maneuver model into an inertial delta-v vector.

    Supports direct impulsive vectors and higher-level intent maneuvers
    (raise perigee, raise semimajor axis, change inclination).

    Args:
        model: Maneuver ``model`` mapping from the scenario file.
        state: Orekit ``SpacecraftState`` at trigger time.
        trigger_type: Trigger type that fired the maneuver.

    Returns:
        Orekit ``Vector3D`` inertial delta-v in m/s.
    """
    if not isinstance(model, Mapping):
        raise TypeError("maneuver.model must be a mapping.")

    model_type = str(model.get("type", "impulsive")).strip().lower()
    if model_type == "impulsive":
        dv_values = to_vector_tuple(model.get("dv_mps"), key_name="model.dv_mps")
        frame_name = str(model.get("frame", "TNW")).strip().upper()
        return local_to_inertial_delta_v(state, dv_values, frame_name)

    if model_type in {
        "intent",
        "raise_perigee",
        "raise_semimajor_axis",
        "maintain_semimajor_axis_above",
        "change_inclination",
    }:
        intent = str(model.get("intent", model_type)).strip().lower()
        if intent == "raise_perigee":
            if trigger_type != "apogee":
                raise ValueError("intent 'raise_perigee' currently requires trigger.type='apogee'.")
            return intent_raise_perigee(model, state)
        if intent in {"raise_semimajor_axis", "maintain_semimajor_axis_above"}:
            return intent_raise_semimajor_axis(model, state)
        if intent == "change_inclination":
            if trigger_type not in {"ascending_node", "descending_node", "epoch"}:
                raise ValueError(
                    "intent 'change_inclination' requires trigger.type in "
                    "{'ascending_node', 'descending_node', 'epoch'}."
                )
            return intent_change_inclination(model, state)
        raise ValueError(
            "Unsupported intent maneuver. Supported intents: "
            "{'raise_perigee', 'raise_semimajor_axis', 'maintain_semimajor_axis_above', 'change_inclination'}."
        )

    raise ValueError("Unsupported maneuver model.type. Supported: {'impulsive', 'intent'}.")


def intent_raise_perigee(model: Mapping[str, Any], state: Any):
    """Compute an apogee burn to raise perigee.

    Args:
        model: Intent model mapping with ``target_perigee_m`` or ``delta_perigee_m``.
        state: Orekit state at burn time (expected near apogee).

    Returns:
        Orekit ``Vector3D`` inertial delta-v.
    """
    from org.orekit.orbits import KeplerianOrbit

    target_defined = "target_perigee_m" in model
    delta_defined = "delta_perigee_m" in model
    if target_defined == delta_defined:
        raise ValueError(
            "intent 'raise_perigee' requires exactly one of model.target_perigee_m or model.delta_perigee_m."
        )

    kep = KeplerianOrbit(state.getOrbit())
    mu = float(kep.getMu())
    a = float(kep.getA())
    e = float(kep.getE())
    ra = a * (1.0 + e)
    rp = a * (1.0 - e)
    target_rp = float(model["target_perigee_m"]) if target_defined else (rp + float(model["delta_perigee_m"]))

    if target_rp <= 0.0:
        raise ValueError("model.target_perigee_m must be positive.")
    if target_rp <= rp:
        raise ValueError("model.target_perigee_m must be greater than current perigee for raise_perigee intent.")
    if target_rp > ra:
        raise ValueError("model.target_perigee_m cannot exceed current apogee for a single apogee burn.")

    v_current = math.sqrt(mu * ((2.0 / ra) - (1.0 / a)))
    a_target = 0.5 * (ra + target_rp)
    v_target = math.sqrt(mu * ((2.0 / ra) - (1.0 / a_target)))
    delta_t = v_target - v_current
    return local_to_inertial_delta_v(state, (delta_t, 0.0, 0.0), "TNW")


def intent_raise_semimajor_axis(model: Mapping[str, Any], state: Any):
    """Compute a tangential burn to raise semimajor axis.

    Args:
        model: Intent model mapping with ``target_a_m``/``delta_a_m`` and
            optional ``min_a_m`` guard shortcut.
        state: Orekit state at burn time.

    Returns:
        Orekit ``Vector3D`` inertial delta-v (zero vector when already above
        ``min_a_m``).
    """
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(state.getOrbit())
    a_now = float(kep.getA())
    mu = float(kep.getMu())

    min_a = model.get("min_a_m")
    if min_a is not None and a_now >= float(min_a):
        return Vector3D(0.0, 0.0, 0.0)

    target_defined = "target_a_m" in model
    delta_defined = "delta_a_m" in model
    if target_defined == delta_defined:
        raise ValueError(
            "intent 'raise_semimajor_axis' requires exactly one of model.target_a_m or model.delta_a_m."
        )

    target_a = float(model["target_a_m"]) if target_defined else (a_now + float(model["delta_a_m"]))
    if target_a <= a_now:
        raise ValueError("target semimajor axis must be greater than current semimajor axis for raise intent.")

    pv = state.getPVCoordinates()
    r_norm = float(pv.getPosition().getNorm())
    v_now = float(pv.getVelocity().getNorm())
    inside = mu * ((2.0 / r_norm) - (1.0 / target_a))
    if inside <= 0.0:
        raise ValueError("Requested target semimajor axis is not reachable at this position.")
    v_target = math.sqrt(inside)
    delta_t = v_target - v_now
    return local_to_inertial_delta_v(state, (delta_t, 0.0, 0.0), "TNW")


def intent_change_inclination(model: Mapping[str, Any], state: Any):
    """Compute an instantaneous plane-change delta-v.

    Args:
        model: Intent model mapping with ``target_i_deg`` or ``delta_i_deg``.
        state: Orekit state at burn time.

    Returns:
        Orekit ``Vector3D`` inertial delta-v.
    """
    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(state.getOrbit())
    i_now_deg = math.degrees(float(kep.getI()))

    if "target_i_deg" in model:
        delta_i_deg = float(model["target_i_deg"]) - i_now_deg
    elif "delta_i_deg" in model:
        delta_i_deg = float(model["delta_i_deg"])
    else:
        raise ValueError("intent 'change_inclination' requires model.target_i_deg or model.delta_i_deg.")

    if abs(delta_i_deg) < 1.0e-12:
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        return Vector3D(0.0, 0.0, 0.0)

    pv = state.getPVCoordinates()
    pos = pv.getPosition()
    vel = pv.getVelocity()
    axis = unit(pos)
    rotated = rotate_vector_about_axis(vel, axis, math.radians(delta_i_deg))
    return rotated.add(vel.scalarMultiply(-1.0))
