"""Covariance and Jacobian transforms for uncertainty propagation."""

from __future__ import annotations

from typing import Any

import numpy as np

from astrodyn_core.uncertainty.matrix_io import (
    java_double_2d_to_numpy,
    new_java_double_2d,
    numpy_to_realmatrix,
    realmatrix_to_numpy,
)


def change_covariance_type(
    cov_6x6: np.ndarray,
    orbit: Any,
    epoch: Any,
    frame: Any,
    from_orbit_type: Any,
    from_pa_type: Any,
    to_orbit_type: Any,
    to_pa_type: Any,
) -> np.ndarray:
    """Use ``StateCovariance.changeCovarianceType`` to re-parametrise a 6×6 covariance."""
    from org.orekit.propagation import StateCovariance

    sc = StateCovariance(
        numpy_to_realmatrix(cov_6x6),
        epoch,
        frame,
        from_orbit_type,
        from_pa_type,
    )
    sc_new = sc.changeCovarianceType(orbit, to_orbit_type, to_pa_type)
    return realmatrix_to_numpy(sc_new.getMatrix())


def orekit_orbit_type(name: str) -> Any:
    """Map string orbit type to Orekit ``OrbitType``."""
    from org.orekit.orbits import OrbitType

    mapping = {
        "CARTESIAN": OrbitType.CARTESIAN,
        "KEPLERIAN": OrbitType.KEPLERIAN,
        "EQUINOCTIAL": OrbitType.EQUINOCTIAL,
    }
    return mapping[name]


def orekit_position_angle(name: str) -> Any:
    """Map string position-angle type to Orekit ``PositionAngleType``."""
    from org.orekit.orbits import PositionAngleType

    mapping = {
        "MEAN": PositionAngleType.MEAN,
        "TRUE": PositionAngleType.TRUE,
        "ECCENTRIC": PositionAngleType.ECCENTRIC,
    }
    return mapping[name]


def configure_cartesian_propagation_basis(propagator: Any) -> None:
    """Force integrated orbit parameters to Cartesian for STM consistency."""
    from org.orekit.orbits import OrbitType, PositionAngleType

    if not hasattr(propagator, "setOrbitType"):
        raise TypeError(
            "STM covariance propagation requires a propagator supporting setOrbitType()."
        )
    propagator.setOrbitType(OrbitType.CARTESIAN)
    if hasattr(propagator, "setPositionAngleType"):
        propagator.setPositionAngleType(PositionAngleType.TRUE)


def orbit_jacobian(
    orbit: Any,
    *,
    from_orbit_type: Any,
    from_pa_type: Any,
    to_orbit_type: Any,
    to_pa_type: Any,
) -> np.ndarray:
    """Return Jacobian ``J = d(to_params) / d(from_params)`` for 6 orbital params."""
    cart_type = orekit_orbit_type("CARTESIAN")
    if (
        from_orbit_type == to_orbit_type
        and (from_orbit_type == cart_type or from_pa_type == to_pa_type)
    ):
        return np.eye(6, dtype=np.float64)

    jac_to_wrt_cart = np.eye(6, dtype=np.float64)
    if to_orbit_type != cart_type:
        orbit_to = to_orbit_type.convertType(orbit)
        j = new_java_double_2d(6, 6)
        orbit_to.getJacobianWrtCartesian(to_pa_type, j)
        jac_to_wrt_cart = java_double_2d_to_numpy(j, 6)

    jac_cart_wrt_from = np.eye(6, dtype=np.float64)
    if from_orbit_type != cart_type:
        orbit_from = from_orbit_type.convertType(orbit)
        j = new_java_double_2d(6, 6)
        orbit_from.getJacobianWrtParameters(from_pa_type, j)
        jac_cart_wrt_from = java_double_2d_to_numpy(j, 6)

    return jac_to_wrt_cart @ jac_cart_wrt_from


def frame_jacobian(from_frame: Any, to_frame: Any, epoch: Any) -> np.ndarray:
    """Return Cartesian PV Jacobian ``d(PV_to) / d(PV_from)`` for a frame transform."""
    if from_frame == to_frame:
        return np.eye(6, dtype=np.float64)

    from org.orekit.utils import CartesianDerivativesFilter

    transform = from_frame.getTransformTo(to_frame, epoch)
    j = new_java_double_2d(6, 6)
    transform.getJacobian(CartesianDerivativesFilter.USE_PV, j)
    return java_double_2d_to_numpy(j, 6)


def transform_covariance_with_jacobian(cov: np.ndarray, jac6: np.ndarray) -> np.ndarray:
    """Apply a 6D Jacobian to 6x6 or 7x7 covariance (mass is unchanged for 7x7)."""
    n = cov.shape[0]
    if n == 6:
        out6 = jac6 @ cov @ jac6.T
        return 0.5 * (out6 + out6.T)
    if n != 7:
        raise ValueError(f"Unsupported covariance shape for Jacobian transform: {cov.shape}.")

    out = np.zeros((7, 7), dtype=np.float64)
    out[:6, :6] = jac6 @ cov[:6, :6] @ jac6.T
    out[:6, 6] = jac6 @ cov[:6, 6]
    out[6, :6] = cov[6, :6] @ jac6.T
    out[6, 6] = float(cov[6, 6])
    return 0.5 * (out + out.T)
