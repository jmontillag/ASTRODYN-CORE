"""Full Lara-Brouwer analytical propagation pipeline.

Initialization -> mean-element propagation -> short-period reconstruction.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval

from .coordinates import (
    cartesian_to_keplerian,
    delaunay_to_keplerian,
    keplerian_to_delaunay,
    polar_to_cartesian,
)
from .mean_elements import propagate_mean_delaunay
from .short_period import (
    brouwer_sp_polar_batch,
    j2_sp_polar_batch,
    mean_to_cartesian_w1_batch,
    osculating_to_mean,
    osculating_to_mean_w1,
    precompute_orbit_averages,
)


class LaraBrouwerPropagator:
    """Brouwer/Lara analytical propagator for J2-J5 zonal harmonics.

    Pipeline:
    1. Initialize: osculating Cartesian -> mean Keplerian -> energy calibration
    2. Propagate: mean Delaunay ODE (J2 secular + J3/J5 frozen) -> mean Keplerian
    3. Reconstruct: mean -> osculating via J2 polar-nodal SP + J3-J5 radial SP -> Cartesian

    J4 secular is excluded because without matching J4 short-period
    corrections, it degrades accuracy (verified numerically).
    """

    def __init__(self, mu, Re, j_coeffs, use_w1_sp=False):
        self.mu = mu
        self.Re = Re
        self.j_coeffs = j_coeffs  # {2: J2, 3: J3, 4: J4, 5: J5}
        self.use_w1_sp = use_w1_sp  # Use W₁ Poisson bracket SP corrections
        self.mean_kep_0 = None
        self.mean_del_0 = None
        self.t0 = None
        self._orbit_averages = None  # Pre-computed <R_n> for SP corrections

    def initialize(self, r0_vec, v0_vec, t0):
        """Initialize from osculating Cartesian state."""
        r0_vec = np.asarray(r0_vec, dtype=float)
        v0_vec = np.asarray(v0_vec, dtype=float)

        # 1. Cartesian -> osculating Keplerian
        osc_kep = cartesian_to_keplerian(r0_vec, v0_vec, self.mu)

        # 2. Osculating -> mean (iterative first-order inverse)
        #    Use W₁-consistent inversion when W₁ SP is enabled.
        from .short_period import osculating_to_mean_w1
        if self.use_w1_sp:
            self.mean_kep_0 = osculating_to_mean_w1(
                osc_kep, self.mu, self.Re, self.j_coeffs[2])
        else:
            self.mean_kep_0 = osculating_to_mean(
                osc_kep, self.mu, self.Re, self.j_coeffs)

        # 3. Convert to Delaunay for ODE integration
        self.mean_del_0 = keplerian_to_delaunay(*self.mean_kep_0, self.mu)
        self.t0 = t0

        # 4. Breakwell-Vagners energy calibration: compute BV-corrected
        #    mean motion offset (stored as dl_bv_correction).
        #    The correction is applied to the secular rate, NOT to the IC.
        self._dl_bv_correction = self._compute_bv_correction(r0_vec, v0_vec)

        # 5. Pre-compute orbit-averaged <R_n> for J3-J5 SP corrections
        a0, e0, i0, Om0, om0, M0 = self.mean_kep_0
        self._orbit_averages = precompute_orbit_averages(
            a0, e0, i0, om0, self.mu, self.Re, self.j_coeffs, n_quad=128)

    def _compute_bv_correction(self, r0, v0):
        """Breakwell-Vagners calibration (Lara 2021 Eq. 23, page 14-15).

        Computes a correction to the secular mean-anomaly rate (dl/dt)
        that accounts for the mismatch between the first-order inverse L'
        and the calibrated L̂.

        The BV procedure:
        1. Compute L̂ from Eq. 23: only the Keplerian term is solved.
        2. The secular rates use n̂ = μ²/L̂³ instead of n' = μ²/L'³.
        3. The correction δn = μ²/L̂³ - μ²/L'³ is added to dl/dt.

        The perturbation-rate terms ∂H₀,m/∂L' are evaluated at L', not L̂,
        so only the Keplerian mean motion is corrected.

        Returns δn (correction to dl/dt in rad/s).
        """
        from .mean_elements import averaged_hamiltonian_H01, averaged_hamiltonian_H02

        r = np.linalg.norm(r0)
        v = np.linalg.norm(v0)

        # Exact total energy including zonal potential
        E0 = 0.5 * v**2 - self.mu / r
        sin_phi = r0[2] / r
        for n_deg, Jn in self.j_coeffs.items():
            coeffs = np.zeros(n_deg + 1)
            coeffs[n_deg] = 1.0
            Pn = legval(sin_phi, coeffs)
            E0 += (self.mu / r) * Jn * (self.Re / r) ** n_deg * Pn

        # Mean Delaunay momenta from the first-order inverse
        ell0, g0, h0, L_bar, G_bar, H_bar = self.mean_del_0
        J2 = self.j_coeffs[2]

        # Evaluate perturbation Hamiltonian terms at the ORIGINAL L', G', H
        H01 = averaged_hamiltonian_H01(L_bar, G_bar, H_bar, self.mu, self.Re)
        H02 = averaged_hamiltonian_H02(L_bar, G_bar, H_bar, self.mu, self.Re)
        sum_Hm = J2 * H01 + 0.5 * J2**2 * H02

        # Direct formula (Eq. 23): solve only the Keplerian term for L̂
        denom = -E0 + sum_Hm
        if abs(denom) > 1e-20:
            L_hat = self.mu / np.sqrt(2.0 * denom)
            # BV correction: difference in Keplerian mean motion
            n_hat = self.mu**2 / L_hat**3
            n_bar = self.mu**2 / L_bar**3
            return n_hat - n_bar

        return 0.0

    def propagate(self, t_array):
        """Propagate to array of times.

        Returns (N,3) positions, (N,3) velocities in km, km/s.
        """
        t_array = np.asarray(t_array, dtype=float)

        # 1. Propagate mean Delaunay elements (with BV mean-motion correction)
        mean_del = propagate_mean_delaunay(
            np.array(self.mean_del_0),
            t_array,
            self.mu,
            self.Re,
            self.j_coeffs,
            dl_bv_correction=self._dl_bv_correction,
        )

        # 2. Convert Delaunay -> Keplerian
        N = len(t_array)
        a_arr = mean_del[:, 3] ** 2 / self.mu   # L^2 / mu
        e_sq = 1.0 - (mean_del[:, 4] / mean_del[:, 3]) ** 2
        e_arr = np.sqrt(np.maximum(e_sq, 0.0))
        cos_i = np.clip(mean_del[:, 4 + 1] / mean_del[:, 4], -1.0, 1.0)
        inc_arr = np.arccos(cos_i)
        Om_arr = mean_del[:, 2]   # h
        om_arr = mean_del[:, 1]   # g
        M_arr = mean_del[:, 0]    # ell

        # 3. Mean -> osculating Cartesian
        J2 = self.j_coeffs[2]

        if self.use_w1_sp:
            # Use exact W₁ Poisson bracket SP corrections (Lara 2021)
            positions, velocities = mean_to_cartesian_w1_batch(
                a_arr, e_arr, inc_arr, Om_arr, om_arr, M_arr,
                self.mu, self.Re, J2,
            )
        else:
            # Legacy polar-nodal SP corrections (SGP4-style)
            r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, inc_osc = brouwer_sp_polar_batch(
                a_arr, e_arr, inc_arr, Om_arr, om_arr, M_arr,
                self.mu, self.Re, J2,
                j_coeffs=self.j_coeffs,
                orbit_averages=self._orbit_averages,
            )

            positions, velocities = polar_to_cartesian(
                r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, inc_osc,
            )

        return positions, velocities
