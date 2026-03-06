"""Composite perturbation model combining multiple sources.

Separates conservative (U) and non-conservative (P) contributions
as required by the GEqOE equations of motion.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel


class CompositePerturbation:
    """Combine multiple perturbation models.

    Conservative models contribute to U (affecting h, c definitions).
    Non-conservative models contribute to P (external forcing only).

    Args:
        conservative: models providing U_expr (e.g., J2). Typically one.
        non_conservative: models providing P_expr (e.g., Sun, Moon, drag).
    """

    def __init__(
        self,
        conservative: list[PerturbationModel] | None = None,
        non_conservative: list | None = None,
    ):
        self.conservative = conservative or []
        self.non_conservative = non_conservative or []

        self.is_conservative = len(self.non_conservative) == 0
        self.is_time_dependent = any(
            getattr(m, "is_time_dependent", False)
            for m in self.conservative + self.non_conservative
        )
        self._force_general = any(
            getattr(m, "_force_general", False)
            for m in self.conservative + self.non_conservative
        )
        self._j2_fast_path = (
            len(self.non_conservative) == 0
            and len(self.conservative) == 1
            and getattr(self.conservative[0], "_j2_fast_path", False)
        )

        # Expose mu and A from the first conservative model (for integrator)
        if self.conservative:
            from astrodyn_core.geqoe_taylor.constants import MU

            first = self.conservative[0]
            self.mu = getattr(first, "mu", MU)
            self.A = getattr(first, "A", 0.0)
        else:
            from astrodyn_core.geqoe_taylor.constants import MU, A_J2
            self.mu = MU
            self.A = A_J2

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        """Sum of conservative potentials."""
        total = 0.0
        for m in self.conservative:
            total = total + m.U_expr(x, y, z, r_mag, t, pars)
        return total

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        """Sum of conservative potentials (numeric)."""
        return sum(m.U_numeric(r_vec, t) for m in self.conservative)

    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple:
        """Sum of conservative potential gradients."""
        total_x, total_y, total_z = 0.0, 0.0, 0.0
        for m in self.conservative:
            dx, dy, dz = m.grad_U_expr(x, y, z, r_mag, t, pars)
            total_x = total_x + dx
            total_y = total_y + dy
            total_z = total_z + dz
        return total_x, total_y, total_z

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple:
        """Sum of non-conservative accelerations."""
        total_x, total_y, total_z = 0.0, 0.0, 0.0
        for m in self.non_conservative:
            px, py, pz = m.P_expr(x, y, z, vx, vy, vz, r_mag, t, pars)
            total_x = total_x + px
            total_y = total_y + py
            total_z = total_z + pz
        return total_x, total_y, total_z

    def U_t_expr(self, x, y, z, r_mag, t, pars: dict):
        """Sum of time derivatives of conservative potentials."""
        total = 0.0
        for m in self.conservative:
            if hasattr(m, "U_t_expr"):
                total = total + m.U_t_expr(x, y, z, r_mag, t, pars)
        return total
