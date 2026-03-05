"""GEqOE Taylor propagator using heyoka.py automatic differentiation.

State vector: [nu, p1, p2, K, q1, q2] in km/s units.
K = generalized eccentric longitude (no Kepler solve needed in the RHS).

Reference: Baù, Hernando-Ayuso & Bombardelli (2021), Celest. Mech. Dyn. Astr. 133:50.
"""

from astrodyn_core.geqoe_taylor.constants import MU, J2, RE, A_J2
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.integrator import (
    build_state_integrator,
    build_stm_integrator,
    propagate,
    extract_stm,
)

__all__ = [
    "MU", "J2", "RE", "A_J2",
    "cart2geqoe", "geqoe2cart",
    "J2Perturbation",
    "build_state_integrator", "build_stm_integrator",
    "propagate", "extract_stm",
]
