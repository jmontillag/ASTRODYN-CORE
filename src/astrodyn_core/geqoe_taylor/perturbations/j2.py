"""J2 zonal harmonic perturbation model.

U = -A/r^3 * (1 - 3*zhat^2)   where A = mu*J2*Re^2/2, zhat = z/r.
Reference: Bau et al. (2021), Eq. 56.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.geqoe_taylor.constants import MU, J2, RE


class J2Perturbation:
    """J2 perturbation model with symbolic and numeric potential."""

    is_conservative = True
    is_time_dependent = False
    _j2_fast_path = True

    def __init__(self, mu: float = MU, j2: float = J2, re: float = RE):
        self.mu = mu
        self.j2 = j2
        self.re = re
        self.A = mu * j2 * re**2 / 2

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        """J2 potential as a heyoka expression.

        U = -A/r^3 * (1 - 3*(z/r)^2)
        """
        if "A_J2" in pars:
            A = pars["A_J2"]
        else:
            A = self.A

        r3 = r_mag * r_mag * r_mag
        zhat2 = (z / r_mag) ** 2
        return -A / r3 * (1.0 - 3.0 * zhat2)

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        """Evaluate J2 potential numerically."""
        r = np.linalg.norm(r_vec)
        zhat = r_vec[2] / r
        return -self.A / r**3 * (1.0 - 3.0 * zhat**2)

    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple:
        """Gradient of J2 potential: (dU/dx, dU/dy, dU/dz).

        dU/dx = 3A*x/r^5 * (1 - 5*z^2/r^2)
        dU/dy = 3A*y/r^5 * (1 - 5*z^2/r^2)
        dU/dz = 3A*z/r^5 * (3 - 5*z^2/r^2)
        """
        if "A_J2" in pars:
            A = pars["A_J2"]
        else:
            A = self.A

        r2 = r_mag * r_mag
        r5 = r2 * r2 * r_mag
        zhat2 = (z * z) / r2
        coeff = 3.0 * A / r5

        dUdx = coeff * x * (1.0 - 5.0 * zhat2)
        dUdy = coeff * y * (1.0 - 5.0 * zhat2)
        dUdz = coeff * z * (3.0 - 5.0 * zhat2)
        return dUdx, dUdy, dUdz

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple:
        """No non-conservative forces for J2."""
        return 0.0, 0.0, 0.0

    def U_t_expr(self, x, y, z, r_mag, t, pars: dict):
        """J2 is time-independent."""
        return 0.0
