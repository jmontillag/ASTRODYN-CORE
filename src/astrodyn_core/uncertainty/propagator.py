"""Backward-compatible facade for uncertainty propagation APIs.

.. deprecated::
    This module is a backward-compatibility facade from Phase B.  All public
    symbols have been moved to their canonical modules:

    - ``uncertainty.stm``        : STMCovariancePropagator
    - ``uncertainty.factory``    : create_covariance_propagator, setup_stm_propagator
    - ``uncertainty.transforms`` : change_covariance_type, orbit_jacobian, ...
    - ``uncertainty.matrix_io``  : realmatrix_to_numpy, numpy_to_realmatrix, ...
    - ``uncertainty.records``    : state_to_orbit_record, numpy_to_nested_tuple

    Import from the canonical modules or use ``UncertaintyClient`` instead.
    This facade and its private underscore aliases will be removed in a
    future release.
"""

from __future__ import annotations

import warnings as _warnings

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


# ---------------------------------------------------------------------------
# Deprecated private aliases — will be removed in a future release.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    _DEPRECATED_ALIASES = {
        "_realmatrix_to_numpy": realmatrix_to_numpy,
        "_new_java_double_2d": new_java_double_2d,
        "_java_double_2d_to_numpy": java_double_2d_to_numpy,
        "_numpy_to_realmatrix": numpy_to_realmatrix,
        "_change_covariance_type": change_covariance_type,
        "_orekit_orbit_type": orekit_orbit_type,
        "_orekit_position_angle": orekit_position_angle,
        "_configure_cartesian_propagation_basis": configure_cartesian_propagation_basis,
        "_orbit_jacobian": orbit_jacobian,
        "_frame_jacobian": frame_jacobian,
        "_transform_covariance_with_jacobian": transform_covariance_with_jacobian,
        "_numpy_to_nested_tuple": numpy_to_nested_tuple,
        "_state_to_orbit_record": state_to_orbit_record,
    }
    if name in _DEPRECATED_ALIASES:
        _warnings.warn(
            f"Importing '{name}' from 'astrodyn_core.uncertainty.propagator' is deprecated. "
            f"Import '{name.lstrip('_')}' from its canonical module instead. "
            "This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module 'astrodyn_core.uncertainty.propagator' has no attribute {name!r}")


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
