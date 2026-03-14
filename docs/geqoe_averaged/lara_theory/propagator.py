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
    j2_sp_polar_batch,
    osculating_to_mean,
)


class LaraBrouwerPropagator:
    """First-order Brouwer/Lara analytical propagator for J2-J5 zonal harmonics.

    Pipeline:
    1. Initialize: osculating Cartesian -> mean Keplerian -> energy calibration
    2. Propagate: mean Delaunay ODE -> mean Keplerian at each epoch
    3. Reconstruct: mean -> osculating via polar-nodal SP -> Cartesian
    """

    def __init__(self, mu, Re, j_coeffs):
        self.mu = mu
        self.Re = Re
        self.j_coeffs = j_coeffs  # {2: J2, 3: J3, 4: J4, 5: J5}
        self.mean_kep_0 = None
        self.mean_del_0 = None
        self.t0 = None

    def initialize(self, r0_vec, v0_vec, t0):
        """Initialize from osculating Cartesian state."""
        r0_vec = np.asarray(r0_vec, dtype=float)
        v0_vec = np.asarray(v0_vec, dtype=float)

        # 1. Cartesian -> osculating Keplerian
        osc_kep = cartesian_to_keplerian(r0_vec, v0_vec, self.mu)

        # 2. Osculating -> mean (iterative first-order inverse)
        self.mean_kep_0 = osculating_to_mean(osc_kep, self.mu, self.Re, self.j_coeffs)

        # 3. Breakwell-Vagners energy calibration
        self._calibrate_energy(r0_vec, v0_vec)

        # 4. Convert to Delaunay for ODE integration
        self.mean_del_0 = keplerian_to_delaunay(*self.mean_kep_0, self.mu)
        self.t0 = t0

    def _calibrate_energy(self, r0, v0):
        """Breakwell-Vagners calibration (Lara 2021 Eq. 23).

        Replaces mean a with calibrated value from exact energy conservation.
        """
        r = np.linalg.norm(r0)
        v = np.linalg.norm(v0)

        # Exact total energy including zonal potential
        E0 = 0.5 * v**2 - self.mu / r
        sin_phi = r0[2] / r
        for n_deg, Jn in self.j_coeffs.items():
            coeffs = np.zeros(n_deg + 1)
            coeffs[n_deg] = 1.0
            Pn = legval(sin_phi, coeffs)
            # Zonal potential U_n = (mu/r)*Jn*(Re/r)^n*Pn (from U = -mu/r*[1 - sum])
            # Total energy E = T + U = 0.5v^2 - mu/r + sum U_n
            E0 += (self.mu / r) * Jn * (self.Re / r) ** n_deg * Pn

        # Averaged J2 Hamiltonian at mean elements
        a_bar = self.mean_kep_0[0]
        e_bar = self.mean_kep_0[1]
        i_bar = self.mean_kep_0[2]
        eta_bar = np.sqrt(1.0 - e_bar**2)
        p_bar = a_bar * (1.0 - e_bar**2)
        s_bar = np.sin(i_bar)

        J2 = self.j_coeffs[2]
        H01 = -(self.mu / (2.0 * a_bar)) * J2 * (self.Re / p_bar) ** 2 * eta_bar * (
            1.0 - 1.5 * s_bar**2
        )

        # Calibrated semi-major axis: solve -mu/(2a_cal) + J2*H01 = E0
        denom = -E0 + H01
        if abs(denom) > 1e-20:
            a_cal = self.mu / (2.0 * denom)
            if a_cal > 0:
                self.mean_kep_0 = (a_cal, *self.mean_kep_0[1:])

    def propagate(self, t_array):
        """Propagate to array of times.

        Returns (N,3) positions, (N,3) velocities in km, km/s.
        """
        t_array = np.asarray(t_array, dtype=float)

        # 1. Propagate mean Delaunay elements
        mean_del = propagate_mean_delaunay(
            np.array(self.mean_del_0),
            t_array,
            self.mu,
            self.Re,
            self.j_coeffs,
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

        # 3. Mean -> osculating Cartesian via polar-nodal SP corrections
        J2 = self.j_coeffs[2]
        r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, inc_osc = j2_sp_polar_batch(
            a_arr, e_arr, inc_arr, Om_arr, om_arr, M_arr,
            self.mu, self.Re, J2,
        )

        positions, velocities = polar_to_cartesian(
            r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, inc_osc,
        )

        return positions, velocities
