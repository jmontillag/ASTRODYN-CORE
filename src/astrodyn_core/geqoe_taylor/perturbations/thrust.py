"""Continuous-thrust perturbation wrapper for GEqOE Taylor propagation."""

from __future__ import annotations

import heyoka as hy
import numpy as np

from astrodyn_core.geqoe_taylor.constants import G0_MPS2, METERS_PER_KILOMETER, MU
from astrodyn_core.geqoe_taylor.thrust import ContinuousThrustLaw


class ContinuousThrustPerturbation:
    """Map a continuous-thrust law into GEqOE-compatible non-conservative forcing.

    The wrapped law returns thrust force components in the RTN frame. This
    perturbation converts them into Cartesian acceleration for the existing
    non-conservative ``P`` term and augments the dynamics with mass depletion.
    """

    is_conservative = False
    requires_mass = True

    def __init__(
        self,
        law: ContinuousThrustLaw,
        name: str = "thrust",
        mu: float = MU,
    ):
        self.law = law
        self.name = name
        self.mu = mu
        self.A = 0.0
        self.is_time_dependent = getattr(law, "is_time_dependent", False)

    def parameter_defaults(self) -> dict[str, float]:
        return dict(self.law.parameter_defaults(self.name))

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        return 0.0

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        del r_vec, t
        return 0.0

    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple:
        return 0.0, 0.0, 0.0

    def _control_state(
        self,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        r_mag,
        t,
        pars: dict,
    ) -> dict[str, object]:
        if "mass" not in pars:
            raise ValueError(
                "Continuous thrust requires a propagated mass state. "
                "Use build_thrust_state_integrator() or build_thrust_stm_integrator()."
            )

        return {
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "r_mag": r_mag,
            "t": t,
            "mass": pars["mass"],
            "K": pars.get("K"),
        }

    def _rtn_basis_expr(self, x, y, z, vx, vy, vz, r_mag):
        e_r = (x / r_mag, y / r_mag, z / r_mag)
        h_x = y * vz - z * vy
        h_y = z * vx - x * vz
        h_z = x * vy - y * vx
        h_mag = hy.sqrt(h_x * h_x + h_y * h_y + h_z * h_z)
        e_n = (h_x / h_mag, h_y / h_mag, h_z / h_mag)
        e_t = (
            e_n[1] * e_r[2] - e_n[2] * e_r[1],
            e_n[2] * e_r[0] - e_n[0] * e_r[2],
            e_n[0] * e_r[1] - e_n[1] * e_r[0],
        )
        return e_r, e_t, e_n

    def _force_and_mass_flow_expr(
        self,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        r_mag,
        t,
        pars: dict,
    ) -> tuple:
        state = self._control_state(x, y, z, vx, vy, vz, r_mag, t, pars)
        e_r, e_t, e_n = self._rtn_basis_expr(x, y, z, vx, vy, vz, r_mag)
        T_r, T_t, T_n, T_mag, Isp = self.law.thrust_rtn_expr(
            state, t, pars, self.name
        )

        mass = pars["mass"]
        accel_scale = 1.0 / (METERS_PER_KILOMETER * mass)

        Fx = T_r * e_r[0] + T_t * e_t[0] + T_n * e_n[0]
        Fy = T_r * e_r[1] + T_t * e_t[1] + T_n * e_n[1]
        Fz = T_r * e_r[2] + T_t * e_t[2] + T_n * e_n[2]

        Px = accel_scale * Fx
        Py = accel_scale * Fy
        Pz = accel_scale * Fz
        m_dot = -T_mag / (G0_MPS2 * Isp)
        return Px, Py, Pz, m_dot

    def P_and_mass_flow_expr(
        self,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        r_mag,
        t,
        pars: dict,
    ) -> tuple:
        return self._force_and_mass_flow_expr(x, y, z, vx, vy, vz, r_mag, t, pars)

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple:
        Px, Py, Pz, _ = self._force_and_mass_flow_expr(
            x, y, z, vx, vy, vz, r_mag, t, pars
        )
        return Px, Py, Pz

    def mass_flow_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict):
        _, _, _, m_dot = self._force_and_mass_flow_expr(
            x, y, z, vx, vy, vz, r_mag, t, pars
        )
        return m_dot

    def U_t_expr(self, x, y, z, r_mag, t, pars: dict):
        return 0.0
