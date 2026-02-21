"""Uncertainty propagation API re-exports.

Convenience re-export module.  All symbols originate from their canonical
submodules (``uncertainty.stm``, ``uncertainty.factory``,
``uncertainty.transforms``, ``uncertainty.matrix_io``,
``uncertainty.records``).  For new code, prefer importing from the canonical
modules or using ``UncertaintyClient``.
"""

from __future__ import annotations

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
]
