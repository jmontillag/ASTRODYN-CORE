"""Continuous-thrust control abstractions for GEqOE Taylor propagation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import heyoka as hy


@runtime_checkable
class ContinuousThrustLaw(Protocol):
    """Interface for smooth continuous-thrust command laws.

    A law returns thrust force components in the RTN/orbital frame together with
    the thrust magnitude used for mass depletion and the specific impulse.

    Units:
    - thrust components / magnitude: N
    - specific impulse: s
    """

    is_time_dependent: bool

    def parameter_defaults(self, prefix: str) -> dict[str, float]:
        """Return runtime parameter defaults keyed by the provided prefix."""
        ...

    def thrust_rtn_expr(
        self,
        state: dict[str, object],
        t,
        pars: dict,
        prefix: str,
    ) -> tuple:
        """Return ``(T_r, T_t, T_n, T_mag, Isp)`` as heyoka expressions."""
        ...


def _par_key(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}"


class ConstantRTNThrustLaw:
    """Constant thrust in the RTN frame.

    The thrust vector is constant in the local orbital frame and is exposed
    through heyoka runtime parameters so the same symbolic graph can be reused
    for coefficient sweeps and future parameter sensitivities.
    """

    is_time_dependent = False

    def __init__(
        self,
        thrust_r_newtons: float = 0.0,
        thrust_t_newtons: float = 0.0,
        thrust_n_newtons: float = 0.0,
        isp_s: float = 3000.0,
    ):
        self.thrust_r_newtons = float(thrust_r_newtons)
        self.thrust_t_newtons = float(thrust_t_newtons)
        self.thrust_n_newtons = float(thrust_n_newtons)
        self.isp_s = float(isp_s)

    def parameter_defaults(self, prefix: str) -> dict[str, float]:
        return {
            _par_key(prefix, "r_newtons"): self.thrust_r_newtons,
            _par_key(prefix, "t_newtons"): self.thrust_t_newtons,
            _par_key(prefix, "n_newtons"): self.thrust_n_newtons,
            _par_key(prefix, "isp_s"): self.isp_s,
        }

    def thrust_rtn_expr(
        self,
        state: dict[str, object],
        t,
        pars: dict,
        prefix: str,
    ) -> tuple:
        del state, t
        T_r = pars[_par_key(prefix, "r_newtons")]
        T_t = pars[_par_key(prefix, "t_newtons")]
        T_n = pars[_par_key(prefix, "n_newtons")]
        T_mag = hy.sqrt(T_r * T_r + T_t * T_t + T_n * T_n)
        Isp = pars[_par_key(prefix, "isp_s")]
        return T_r, T_t, T_n, T_mag, Isp
