"""GEqOE first-order averaged theory package.

Public API for the mixed-zonal (J2-J5) first-order Lie-Deprit averaged theory
of Generalized Equinoctial Orbital Elements.
"""

# Coordinate utilities
from .coordinates import kepler_to_rv, rot1, rot3, rv_to_classical

# Shared constants
from .constants import J2, J3, J4, J5, J_COEFFS, MU, RE

# Symbolic engine (degree-n averaged rates)
from .symbolic import (
    evaluate_truncated_mean_rates,
    evaluate_truncated_mean_rhs_pq,
    harmonic_coefficients,
    kernel_core,
    q_from_g,
)

# Short-period map (osculating <-> mean transformations)
from .short_period import (
    evaluate_truncated_mean_rhs_pqm,
    evaluate_truncated_short_period,
    isolated_short_period_expressions_for,
    mean_to_osculating_state,
    osculating_to_mean_state,
)

# Frozen-state numerical averaging
from .fourier_model import avg_slow_drift, fit_total_order_model, frozen_state

__all__ = [
    # coordinates
    "kepler_to_rv",
    "rot1",
    "rot3",
    "rv_to_classical",
    # constants
    "J2",
    "J3",
    "J4",
    "J5",
    "J_COEFFS",
    "MU",
    "RE",
    # symbolic
    "evaluate_truncated_mean_rates",
    "evaluate_truncated_mean_rhs_pq",
    "harmonic_coefficients",
    "kernel_core",
    "q_from_g",
    # short period
    "evaluate_truncated_mean_rhs_pqm",
    "evaluate_truncated_short_period",
    "isolated_short_period_expressions_for",
    "mean_to_osculating_state",
    "osculating_to_mean_state",
    # fourier
    "avg_slow_drift",
    "fit_total_order_model",
    "frozen_state",
]
