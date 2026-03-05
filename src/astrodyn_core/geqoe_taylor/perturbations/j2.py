"""J2 zonal harmonic perturbation model.

U = -A/r^3 * (1 - 3*zhat^2)   where A = mu*J2*Re^2/2, zhat = z/r.
Reference: Baù et al. (2021), Eq. 56.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.geqoe_taylor.constants import MU, J2, RE


class J2Perturbation:
    """J2 perturbation model with symbolic and numeric potential."""

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
        """Evaluate J2 potential numerically.

        Args:
            r_vec: position (3,) in km.

        Returns:
            U in km^2/s^2.
        """
        r = np.linalg.norm(r_vec)
        zhat = r_vec[2] / r
        return -self.A / r**3 * (1.0 - 3.0 * zhat**2)
