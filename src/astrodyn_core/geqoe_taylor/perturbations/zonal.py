"""Zonal harmonic perturbation model (J2 through Jn_max).

Conservative, time-independent geopotential perturbation using Legendre
polynomials for arbitrary zonal harmonics.

Two interfaces:
  - Standard (U_expr, grad_U_expr): Cartesian placeholders + hy.diff_tensors.
    Used by the general path when wrapped in CompositePerturbation.
  - Fast (zonal_quantities): builds U, dU/dzhat, and Euler term directly
    from r and zhat. Used by the dedicated zonal path in rhs.py.

Reference: Bau et al. (2021), Section 7; EGM2008 coefficients.
"""

from __future__ import annotations

import heyoka as hy
import numpy as np

from astrodyn_core.geqoe_taylor.constants import MU, J2, RE


def _legendre_P(n: int, x):
    """Legendre polynomial P_n(x) via Bonnet recurrence.

    Works with both heyoka expressions and numeric (float) values.
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    P_prev, P_curr = 1.0, x
    for k in range(2, n + 1):
        P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
        P_prev, P_curr = P_curr, P_next
    return P_curr


def _legendre_P_and_deriv(n: int, x):
    """Legendre polynomial P_n(x) and its derivative P'_n(x).

    Uses differentiated Bonnet recurrence:
        k*P'_k = (2k-1)*(P_{k-1} + x*P'_{k-1}) - (k-1)*P'_{k-2}

    Works with both heyoka expressions and numeric (float) values.

    Returns:
        (P_n, P'_n)
    """
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0
    P_prev, P_curr = 1.0, x
    dP_prev, dP_curr = 0.0, 1.0
    for k in range(2, n + 1):
        P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
        dP_next = ((2 * k - 1) * (P_curr + x * dP_curr) - (k - 1) * dP_prev) / k
        P_prev, P_curr = P_curr, P_next
        dP_prev, dP_curr = dP_curr, dP_next
    return P_curr, dP_curr


class ZonalPerturbation:
    """Zonal harmonic perturbation J2 through Jn_max.

    Builds U = sum_n mu*Jn*Re^n / r^(n+1) * P_n(z/r) symbolically.

    The fast path (zonal_quantities) computes U, dU/dzhat, and the Euler
    homogeneity term directly from r and zhat — no Cartesian coordinates
    or 3D gradient needed.

    Args:
        j_coeffs: {degree: Jn_value} mapping (e.g., {2: J2, 3: J3, 4: J4}).
        mu: gravitational parameter (km^3/s^2).
        re: reference radius (km).
    """

    is_conservative = True
    is_time_dependent = False
    _zonal_fast_path = True
    _force_general = True  # fallback when wrapped in Composite

    def __init__(
        self,
        j_coeffs: dict[int, float],
        mu: float = MU,
        re: float = RE,
    ):
        if not j_coeffs:
            raise ValueError("j_coeffs must be non-empty")
        if any(n < 2 for n in j_coeffs):
            raise ValueError("Zonal harmonic degree must be >= 2")

        self.j_coeffs = dict(j_coeffs)
        self.mu = mu
        self.re = re
        self.A = mu * j_coeffs.get(2, J2) * re**2 / 2

        # Pre-compute coefficients: Cn = mu * Jn * Re^n
        self._Cn = {n: mu * Jn * re**n for n, Jn in j_coeffs.items()}

        self._cart_grad_cache = None

    # ------------------------------------------------------------------
    # Fast path: used by _build_zonal_system in rhs.py
    # ------------------------------------------------------------------

    def zonal_quantities(self, r, zhat):
        """Compute U, dU/dzhat, and Euler term in a single Legendre pass.

        For zonal harmonics of degree n:
            U_n = C_n / r^(n+1) * P_n(zhat)
            dU_n/dzhat = C_n / r^(n+1) * P'_n(zhat)
            euler_n = (1-n) * U_n          (from Euler homogeneity theorem)

        Returns:
            (U, dU_dzhat, euler_term) as heyoka expressions or floats.
        """
        U = 0.0
        dU_dzhat = 0.0
        euler = 0.0
        for n in sorted(self._Cn):
            Cn = self._Cn[n]
            Pn, dPn = _legendre_P_and_deriv(n, zhat)
            r_inv_np1 = r ** (-(n + 1))
            U_n = Cn * r_inv_np1 * Pn
            U = U + U_n
            dU_dzhat = dU_dzhat + Cn * r_inv_np1 * dPn
            euler = euler + (1 - n) * U_n
        return U, dU_dzhat, euler

    # ------------------------------------------------------------------
    # Standard interface: used by general path and Cowell ground truth
    # ------------------------------------------------------------------

    def _ensure_cart_grad(self):
        """Lazy-build Cartesian gradient via diff_tensors (for general path)."""
        if self._cart_grad_cache is not None:
            return
        _x, _y, _z = hy.make_vars("_zx", "_zy", "_zz")
        _r2 = _x * _x + _y * _y + _z * _z
        _r = hy.sqrt(_r2)
        _zhat = _z / _r

        U = 0.0
        for n in sorted(self._Cn):
            Cn = self._Cn[n]
            Pn = _legendre_P(n, _zhat)
            U = U + Cn / _r ** (n + 1) * Pn

        dt = hy.diff_tensors([U], [_x, _y, _z], diff_order=1)
        self._cart_grad_cache = (U, list(dt.gradient))

    def _smap(self, x, y, z) -> dict:
        return {"_zx": x, "_zy": y, "_zz": z}

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        self._ensure_cart_grad()
        return hy.subs(self._cart_grad_cache[0], self._smap(x, y, z))

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        r = np.linalg.norm(r_vec)
        zhat = r_vec[2] / r
        U, _, _ = self.zonal_quantities(r, zhat)
        return float(U)

    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple:
        self._ensure_cart_grad()
        smap = self._smap(x, y, z)
        return tuple(hy.subs(self._cart_grad_cache[1], smap))

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple:
        return 0.0, 0.0, 0.0

    def U_t_expr(self, x, y, z, r_mag, t, pars: dict):
        return 0.0
