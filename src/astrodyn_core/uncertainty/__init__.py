"""Uncertainty propagation module: covariance via STM.

Public API
----------
UncertaintyClient          Facade for covariance propagation and I/O workflows.
UncertaintySpec            Configuration for uncertainty propagation method.
CovarianceRecord           Single epoch covariance record.
CovarianceSeries           Ordered covariance records with metadata.
STMCovariancePropagator    STM-based covariance propagator.
change_covariance_type     Re-parametrise a 6x6 covariance between orbit types.
create_covariance_propagator  Factory for covariance propagator instances.
setup_stm_propagator       Prepare a propagator for raw STM extraction.
save_covariance_series     Save covariance series (auto-detects YAML/HDF5).
load_covariance_series     Load covariance series (auto-detects YAML/HDF5).
"""

from astrodyn_core.uncertainty.client import UncertaintyClient
from astrodyn_core.uncertainty.io import (
    load_covariance_series,
    save_covariance_series,
)
from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries
from astrodyn_core.uncertainty.factory import (
    create_covariance_propagator,
    setup_stm_propagator,
)
from astrodyn_core.uncertainty.stm import STMCovariancePropagator
from astrodyn_core.uncertainty.spec import UncertaintySpec
from astrodyn_core.uncertainty.transforms import change_covariance_type

__all__ = [
    "UncertaintyClient",
    "UncertaintySpec",
    "CovarianceRecord",
    "CovarianceSeries",
    "STMCovariancePropagator",
    "change_covariance_type",
    "create_covariance_propagator",
    "setup_stm_propagator",
    "save_covariance_series",
    "load_covariance_series",
]
