"""Backward-compatible façade for uncertainty propagation APIs."""

from astrodyn_core.uncertainty.factory import (
    UnscentedCovariancePropagator,
    create_covariance_propagator,
    setup_stm_propagator,
)
from astrodyn_core.uncertainty.matrix_io import (
    java_double_2d_to_numpy,
    new_java_double_2d,
    numpy_to_realmatrix,
    realmatrix_to_numpy,
)
from astrodyn_core.uncertainty.records import numpy_to_nested_tuple, state_to_orbit_record
from astrodyn_core.uncertainty.stm import STMCovariancePropagator
from astrodyn_core.uncertainty.transforms import (
    change_covariance_type,
    configure_cartesian_propagation_basis,
    frame_jacobian,
    orbit_jacobian,
    orekit_orbit_type,
    orekit_position_angle,
    transform_covariance_with_jacobian,
)

# Legacy private aliases kept for backward compatibility.
_realmatrix_to_numpy = realmatrix_to_numpy
_new_java_double_2d = new_java_double_2d
_java_double_2d_to_numpy = java_double_2d_to_numpy
_numpy_to_realmatrix = numpy_to_realmatrix
_change_covariance_type = change_covariance_type
_orekit_orbit_type = orekit_orbit_type
_orekit_position_angle = orekit_position_angle
_configure_cartesian_propagation_basis = configure_cartesian_propagation_basis
_orbit_jacobian = orbit_jacobian
_frame_jacobian = frame_jacobian
_transform_covariance_with_jacobian = transform_covariance_with_jacobian
_numpy_to_nested_tuple = numpy_to_nested_tuple
_state_to_orbit_record = state_to_orbit_record

__all__ = [
    "STMCovariancePropagator",
    "UnscentedCovariancePropagator",
    "create_covariance_propagator",
    "setup_stm_propagator",
    "change_covariance_type",
    "frame_jacobian",
    "orbit_jacobian",
    "transform_covariance_with_jacobian",
    "realmatrix_to_numpy",
    "numpy_to_realmatrix",
    "state_to_orbit_record",
    # legacy aliases
    "_change_covariance_type",
    "_frame_jacobian",
    "_orbit_jacobian",
    "_transform_covariance_with_jacobian",
    "_realmatrix_to_numpy",
    "_numpy_to_realmatrix",
    "_state_to_orbit_record",
]
