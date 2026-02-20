"""Vector and frame helper utilities for mission maneuver computations."""

from __future__ import annotations

import math
from typing import Any, Sequence


def local_to_inertial_delta_v(state: Any, components: tuple[float, float, float], frame_name: str):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    c1, c2, c3 = components
    b1, b2, b3 = local_basis_vectors(state, frame_name)
    return b1.scalarMultiply(c1).add(b2.scalarMultiply(c2)).add(b3.scalarMultiply(c3))


def local_basis_vectors(state: Any, frame_name: str):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    pv = state.getPVCoordinates()
    r = pv.getPosition()
    v = pv.getVelocity()
    w = unit(Vector3D.crossProduct(r, v))

    if frame_name == "TNW":
        t = unit(v)
        n = unit(Vector3D.crossProduct(w, t))
        return t, n, w

    if frame_name == "RTN":
        r_hat = unit(r)
        t_hat = unit(Vector3D.crossProduct(w, r_hat))
        return r_hat, t_hat, w

    if frame_name == "INERTIAL":
        from org.hipparchus.geometry.euclidean.threed import Vector3D as V3

        return V3(1.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), V3(0.0, 0.0, 1.0)

    raise ValueError("Unsupported maneuver frame. Supported: {'TNW', 'RTN', 'INERTIAL'}.")


def rotate_vector_about_axis(vector: Any, axis: Any, angle_rad: float):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    term1 = vector.scalarMultiply(c)
    term2 = Vector3D.crossProduct(axis, vector).scalarMultiply(s)
    term3 = axis.scalarMultiply(float(Vector3D.dotProduct(axis, vector)) * (1.0 - c))
    return term1.add(term2).add(term3)


def unit(vector: Any):
    norm = float(vector.getNorm())
    if norm <= 0.0:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector.scalarMultiply(1.0 / norm)


def to_vector_tuple(values: Any, *, key_name: str) -> tuple[float, float, float]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{key_name} must be a 3-element numeric sequence.")
    if len(values) != 3:
        raise ValueError(f"{key_name} must contain exactly 3 components.")
    return (float(values[0]), float(values[1]), float(values[2]))


def tuple_to_vector(values: Sequence[float]):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    return Vector3D(float(values[0]), float(values[1]), float(values[2]))
