"""Base protocol for perturbation models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PerturbationModel(Protocol):
    """Interface for perturbation models used in the GEqOE propagator.

    A model must provide at minimum:
    - U_expr: heyoka symbolic expression for the disturbing potential U
    - U_numeric: scalar evaluation of U for coordinate conversions

    For the general equations (non-J2-only), models should also provide:
    - grad_U_expr: gradient of U in Cartesian coordinates
    - P_expr: non-conservative acceleration in Cartesian
    - U_t_expr: time derivative of U
    - is_conservative: True if P = 0
    - is_time_dependent: True if U depends on time
    """

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        """Return U as a heyoka symbolic expression."""
        ...

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        """Evaluate U numerically (for coordinate conversions)."""
        ...


@runtime_checkable
class GeneralPerturbationModel(PerturbationModel, Protocol):
    """Extended interface for general (non-J2-only) perturbation models.

    In addition to U_expr/U_numeric, provides force decomposition needed
    for the full GEqOE equations of motion (Eqs. 45-51 of Bau et al. 2021).
    """

    is_conservative: bool
    is_time_dependent: bool

    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple:
        """Gradient of U: (dU/dx, dU/dy, dU/dz) as heyoka expressions."""
        ...

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple:
        """Non-conservative acceleration (Px, Py, Pz) in Cartesian frame."""
        ...

    def U_t_expr(self, x, y, z, r_mag, t, pars: dict):
        """Partial time derivative of U (dU/dt). Zero for static potentials."""
        ...
