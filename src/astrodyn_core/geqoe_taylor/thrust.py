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


def _cubic_hermite_expr(y0, y1, m0, m1, tau):
    """Evaluate a cubic Hermite segment on normalized arc time ``tau``."""
    tau2 = tau * tau
    tau3 = tau2 * tau
    h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
    h10 = tau3 - 2.0 * tau2 + tau
    h01 = -2.0 * tau3 + 3.0 * tau2
    h11 = tau3 - tau2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def _normalize_coefficients(
    values: tuple[float, ...] | list[float] | None,
    order: int,
    *,
    name: str,
) -> tuple[float, ...]:
    if values is None:
        return tuple(0.0 for _ in range(order))
    coeffs = tuple(float(v) for v in values)
    if len(coeffs) != order:
        raise ValueError(f"{name} must have length {order}.")
    return coeffs


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


class FourierKRTNThrustLaw:
    """Direct RTN Fourier series in the propagated generalized longitude ``K``.

    This is a full-dynamics control law: the thrust components are evaluated
    directly inside the GEqOE propagation rather than through any orbit-averaged
    or mean-element reduction.
    """

    is_time_dependent = False

    def __init__(
        self,
        order: int,
        *,
        bias_r_newtons: float = 0.0,
        bias_t_newtons: float = 0.0,
        bias_n_newtons: float = 0.0,
        cosine_r_newtons: tuple[float, ...] | list[float] | None = None,
        cosine_t_newtons: tuple[float, ...] | list[float] | None = None,
        cosine_n_newtons: tuple[float, ...] | list[float] | None = None,
        sine_r_newtons: tuple[float, ...] | list[float] | None = None,
        sine_t_newtons: tuple[float, ...] | list[float] | None = None,
        sine_n_newtons: tuple[float, ...] | list[float] | None = None,
        phase_origin_rad: float = 0.0,
        isp_s: float = 3000.0,
    ):
        if int(order) != order or order < 0:
            raise ValueError("order must be a non-negative integer.")

        self.order = int(order)
        self.bias_r_newtons = float(bias_r_newtons)
        self.bias_t_newtons = float(bias_t_newtons)
        self.bias_n_newtons = float(bias_n_newtons)
        self.cosine_r_newtons = _normalize_coefficients(
            cosine_r_newtons,
            self.order,
            name="cosine_r_newtons",
        )
        self.cosine_t_newtons = _normalize_coefficients(
            cosine_t_newtons,
            self.order,
            name="cosine_t_newtons",
        )
        self.cosine_n_newtons = _normalize_coefficients(
            cosine_n_newtons,
            self.order,
            name="cosine_n_newtons",
        )
        self.sine_r_newtons = _normalize_coefficients(
            sine_r_newtons,
            self.order,
            name="sine_r_newtons",
        )
        self.sine_t_newtons = _normalize_coefficients(
            sine_t_newtons,
            self.order,
            name="sine_t_newtons",
        )
        self.sine_n_newtons = _normalize_coefficients(
            sine_n_newtons,
            self.order,
            name="sine_n_newtons",
        )
        self.phase_origin_rad = float(phase_origin_rad)
        self.isp_s = float(isp_s)

    def _series_parameter_defaults(
        self,
        prefix: str,
        component: str,
        bias: float,
        cosine_coeffs: tuple[float, ...],
        sine_coeffs: tuple[float, ...],
    ) -> dict[str, float]:
        defaults = {_par_key(prefix, f"{component}_bias_newtons"): bias}
        for harmonic, value in enumerate(cosine_coeffs, start=1):
            defaults[_par_key(prefix, f"{component}_cos{harmonic}_newtons")] = value
        for harmonic, value in enumerate(sine_coeffs, start=1):
            defaults[_par_key(prefix, f"{component}_sin{harmonic}_newtons")] = value
        return defaults

    def parameter_defaults(self, prefix: str) -> dict[str, float]:
        defaults = {}
        defaults.update(
            self._series_parameter_defaults(
                prefix,
                "r",
                self.bias_r_newtons,
                self.cosine_r_newtons,
                self.sine_r_newtons,
            )
        )
        defaults.update(
            self._series_parameter_defaults(
                prefix,
                "t",
                self.bias_t_newtons,
                self.cosine_t_newtons,
                self.sine_t_newtons,
            )
        )
        defaults.update(
            self._series_parameter_defaults(
                prefix,
                "n",
                self.bias_n_newtons,
                self.cosine_n_newtons,
                self.sine_n_newtons,
            )
        )
        defaults[_par_key(prefix, "isp_s")] = self.isp_s
        return defaults

    def _component_expr(
        self,
        component: str,
        angle,
        pars: dict,
        prefix: str,
    ):
        expr = pars[_par_key(prefix, f"{component}_bias_newtons")]
        for harmonic in range(1, self.order + 1):
            expr = expr + pars[_par_key(prefix, f"{component}_cos{harmonic}_newtons")] * hy.cos(
                harmonic * angle
            )
            expr = expr + pars[_par_key(prefix, f"{component}_sin{harmonic}_newtons")] * hy.sin(
                harmonic * angle
            )
        return expr

    def thrust_rtn_expr(
        self,
        state: dict[str, object],
        t,
        pars: dict,
        prefix: str,
    ) -> tuple:
        del t
        angle = state["K"] - self.phase_origin_rad

        T_r = self._component_expr("r", angle, pars, prefix)
        T_t = self._component_expr("t", angle, pars, prefix)
        T_n = self._component_expr("n", angle, pars, prefix)
        T_mag = hy.sqrt(T_r * T_r + T_t * T_t + T_n * T_n)
        Isp = pars[_par_key(prefix, "isp_s")]
        return T_r, T_t, T_n, T_mag, Isp


class CubicHermiteRTNThrustLaw:
    """Single-arc cubic Hermite spline in RTN thrust components.

    The law is defined over normalized arc time ``tau = t / duration_s``.
    It is intended for propagation over a single arc with ``0 <= t <= duration_s``.
    Endpoint slopes are derivatives with respect to the normalized time ``tau``,
    so they carry the same units as thrust.
    """

    is_time_dependent = True

    def __init__(
        self,
        duration_s: float,
        thrust_r_newtons: tuple[float, float] = (0.0, 0.0),
        thrust_t_newtons: tuple[float, float] = (0.0, 0.0),
        thrust_n_newtons: tuple[float, float] = (0.0, 0.0),
        slope_r_newtons: tuple[float, float] = (0.0, 0.0),
        slope_t_newtons: tuple[float, float] = (0.0, 0.0),
        slope_n_newtons: tuple[float, float] = (0.0, 0.0),
        isp_s: float = 3000.0,
    ):
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

        self.duration_s = float(duration_s)
        self.thrust_r_newtons = tuple(float(v) for v in thrust_r_newtons)
        self.thrust_t_newtons = tuple(float(v) for v in thrust_t_newtons)
        self.thrust_n_newtons = tuple(float(v) for v in thrust_n_newtons)
        self.slope_r_newtons = tuple(float(v) for v in slope_r_newtons)
        self.slope_t_newtons = tuple(float(v) for v in slope_t_newtons)
        self.slope_n_newtons = tuple(float(v) for v in slope_n_newtons)
        self.isp_s = float(isp_s)

    def parameter_defaults(self, prefix: str) -> dict[str, float]:
        return {
            _par_key(prefix, "r0_newtons"): self.thrust_r_newtons[0],
            _par_key(prefix, "r1_newtons"): self.thrust_r_newtons[1],
            _par_key(prefix, "t0_newtons"): self.thrust_t_newtons[0],
            _par_key(prefix, "t1_newtons"): self.thrust_t_newtons[1],
            _par_key(prefix, "n0_newtons"): self.thrust_n_newtons[0],
            _par_key(prefix, "n1_newtons"): self.thrust_n_newtons[1],
            _par_key(prefix, "r0_slope_newtons"): self.slope_r_newtons[0],
            _par_key(prefix, "r1_slope_newtons"): self.slope_r_newtons[1],
            _par_key(prefix, "t0_slope_newtons"): self.slope_t_newtons[0],
            _par_key(prefix, "t1_slope_newtons"): self.slope_t_newtons[1],
            _par_key(prefix, "n0_slope_newtons"): self.slope_n_newtons[0],
            _par_key(prefix, "n1_slope_newtons"): self.slope_n_newtons[1],
            _par_key(prefix, "isp_s"): self.isp_s,
        }

    def thrust_rtn_expr(
        self,
        state: dict[str, object],
        t,
        pars: dict,
        prefix: str,
    ) -> tuple:
        del state
        tau = t / self.duration_s

        T_r = _cubic_hermite_expr(
            pars[_par_key(prefix, "r0_newtons")],
            pars[_par_key(prefix, "r1_newtons")],
            pars[_par_key(prefix, "r0_slope_newtons")],
            pars[_par_key(prefix, "r1_slope_newtons")],
            tau,
        )
        T_t = _cubic_hermite_expr(
            pars[_par_key(prefix, "t0_newtons")],
            pars[_par_key(prefix, "t1_newtons")],
            pars[_par_key(prefix, "t0_slope_newtons")],
            pars[_par_key(prefix, "t1_slope_newtons")],
            tau,
        )
        T_n = _cubic_hermite_expr(
            pars[_par_key(prefix, "n0_newtons")],
            pars[_par_key(prefix, "n1_newtons")],
            pars[_par_key(prefix, "n0_slope_newtons")],
            pars[_par_key(prefix, "n1_slope_newtons")],
            tau,
        )

        T_mag = hy.sqrt(T_r * T_r + T_t * T_t + T_n * T_n)
        Isp = pars[_par_key(prefix, "isp_s")]
        return T_r, T_t, T_n, T_mag, Isp
