"""GEQOE propagation package.

This package provides the Taylor-series J2 propagator using Generalized
Equinoctial Orbital Elements (GEqOE), with full State Transition Matrix
computation up to 4th order.
"""

from astrodyn_core.propagation.geqoe.core import j2_taylor_propagator, taylor_cart_propagator
from astrodyn_core.propagation.geqoe.state import (
    GEqOEPropagationConstants,
    GEqOEPropagationContext,
    GEqOEState,
)

__all__ = [
    "GEqOEState",
    "GEqOEPropagationConstants",
    "GEqOEPropagationContext",
    "j2_taylor_propagator",
    "taylor_cart_propagator",
]
