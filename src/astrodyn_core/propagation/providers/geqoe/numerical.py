"""Numerical GEqOE propagator — scipy integration of the exact GEqOE ODE.

Integrates the J2-perturbed GEqOE equations of motion using
``scipy.integrate.solve_ivp``.  Intended as a reference implementation
for verifying the Taylor-series propagators: since both share the
**identical** ODE, any disagreement isolates pure Taylor truncation error.

Not intended for operational use (no STM, slower than Taylor).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.geqoe.conversion import BodyConstants, geqoe2rv, rv2geqoe
from astrodyn_core.propagation.geqoe.ode import geqoe_rhs
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.providers.geqoe.propagator import (
    _cartesian_to_orekit_state,
    _check_mu_consistency,
    _orbit_to_state,
    _orekit_state_to_cartesian,
    _resolve_body_constants_from_orekit,
)
from astrodyn_core.propagation.specs import PropagatorSpec

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumericalGEqOEPropagator
# ---------------------------------------------------------------------------


class NumericalGEqOEPropagator:
    """J2 GEqOE propagator via numerical integration (scipy).

    Parameters
    ----------
    initial_orbit : Any
        Orekit ``Orbit`` object representing the initial state.
    body_constants : Mapping[str, float] | None
        ``{"mu", "j2", "re"}`` in SI.  Resolved from Orekit WGS84 when *None*.
    mass_kg : float
        Spacecraft mass in kilograms.
    integrator : str
        scipy ``solve_ivp`` method name (default ``"DOP853"``).
    rtol : float
        Relative tolerance for the integrator.
    atol : float
        Absolute tolerance for the integrator.
    """

    def __init__(
        self,
        initial_orbit: Any,
        body_constants: Mapping[str, float] | None = None,
        mass_kg: float = 1000.0,
        integrator: str = "DOP853",
        rtol: float = 1e-12,
        atol: float = 1e-12,
    ) -> None:
        self._orekit_available = False
        try:
            from org.orekit.propagation import AbstractPropagator  # noqa: F401

            self._orekit_available = True
        except Exception:
            pass

        if body_constants is None:
            body_constants = _resolve_body_constants_from_orekit()

        self._integrator = integrator
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._mass_kg = float(mass_kg)
        self._body_constants = dict(body_constants)
        self._bc = BodyConstants(
            j2=float(body_constants["j2"]),
            re=float(body_constants["re"]),
            mu=float(body_constants["mu"]),
        )
        self._time_scale = float((self._bc.re**3 / self._bc.mu) ** 0.5)
        self._nfev: int = 0
        self._initial_orbit = initial_orbit

        if self._orekit_available and initial_orbit is not None:
            initial_state = _orbit_to_state(initial_orbit, mass_kg)
            self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
                _orekit_state_to_cartesian(initial_state)
            )
            _check_mu_consistency(self._mu, self._bc.mu)
            self._eq0 = self._cart_to_geqoe_norm(self._y0)
        else:
            self._y0 = None
            self._frame = None
            self._epoch = None
            self._mu = float(body_constants["mu"])
            self._eq0 = None

    # ------------------------------------------------------------------
    # GEqOE <-> Cartesian helpers
    # ------------------------------------------------------------------

    def _cart_to_geqoe_norm(self, y0_cart: np.ndarray) -> np.ndarray:
        """Convert SI Cartesian to normalized GEqOE state for the ODE."""
        nu_phys, q1, q2, p1, p2, Lr = rv2geqoe(0.0, y0_cart, self._bc)
        # rv2geqoe returns arrays; extract scalars
        nu_norm = float(nu_phys[0]) * self._time_scale
        return np.array([
            nu_norm,
            float(q1[0]),
            float(q2[0]),
            float(p1[0]),
            float(p2[0]),
            float(Lr[0]),
        ])

    def _geqoe_norm_to_cart(self, eq_states: np.ndarray) -> np.ndarray:
        """Convert normalized GEqOE states (N, 6) to SI Cartesian (N, 6)."""
        eq_phys = eq_states.copy()
        eq_phys[:, 0] /= self._time_scale  # nu_norm -> nu_physical
        rv, rpv = geqoe2rv(0.0, eq_phys, self._bc)
        return np.hstack([rv, rpv])

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def _integrate(self, t_norm_eval: np.ndarray) -> np.ndarray:
        """Integrate the GEqOE ODE and evaluate at given normalized times.

        Returns (N, 6) array of normalized GEqOE states.
        """
        assert self._eq0 is not None
        self._nfev = 0
        N = len(t_norm_eval)
        result = np.zeros((N, 6))

        zero_mask = t_norm_eval == 0.0
        if np.any(zero_mask):
            result[zero_mask] = self._eq0

        # Forward integration
        fwd_mask = t_norm_eval > 0
        if np.any(fwd_mask):
            t_fwd = t_norm_eval[fwd_mask]
            sol = solve_ivp(
                geqoe_rhs,
                [0.0, np.max(t_fwd)],
                self._eq0,
                method=self._integrator,
                rtol=self._rtol,
                atol=self._atol,
                args=(self._bc.j2,),
                dense_output=True,
            )
            if not sol.success:
                raise RuntimeError(f"Forward integration failed: {sol.message}")
            self._nfev += sol.nfev
            result[fwd_mask] = sol.sol(t_fwd).T

        # Backward integration
        bwd_mask = t_norm_eval < 0
        if np.any(bwd_mask):
            t_bwd = t_norm_eval[bwd_mask]
            sol = solve_ivp(
                geqoe_rhs,
                [0.0, np.min(t_bwd)],
                self._eq0,
                method=self._integrator,
                rtol=self._rtol,
                atol=self._atol,
                args=(self._bc.j2,),
                dense_output=True,
            )
            if not sol.success:
                raise RuntimeError(f"Backward integration failed: {sol.message}")
            self._nfev += sol.nfev
            result[bwd_mask] = sol.sol(t_bwd).T

        return result

    # ------------------------------------------------------------------
    # Orekit-compatible interface
    # ------------------------------------------------------------------

    def propagate(self, start: Any, target: Any | None = None) -> Any:
        """Propagate to *target* and return an Orekit ``SpacecraftState``."""
        if target is None:
            target = start
            start = self._epoch

        if not self._orekit_available:
            raise RuntimeError(
                "Orekit is not available. Cannot call propagate() without Orekit."
            )

        dt_seconds = float(target.durationFrom(self._epoch))
        t_norm = np.array([dt_seconds / self._time_scale])
        eq_out = self._integrate(t_norm)
        y_cart = self._geqoe_norm_to_cart(eq_out)

        return _cartesian_to_orekit_state(
            y_cart[0],
            frame=self._frame,
            date=target,
            mu=self._mu,
            mass=self._mass_kg,
        )

    def propagate_array(
        self,
        dt_seconds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch propagation returning raw numpy arrays.

        Args:
            dt_seconds: Time offsets from the initial epoch in seconds.

        Returns:
            Tuple ``(y_out, stm)`` with Cartesian states ``(N, 6)`` and
            identity STMs ``(6, 6, N)`` (STM not supported).
        """
        if self._eq0 is None:
            raise RuntimeError(
                "Initial state not available. "
                "Orekit may not have been initialised when the propagator was created."
            )
        tspan = np.atleast_1d(np.asarray(dt_seconds, dtype=float))
        N = len(tspan)
        t_norm = tspan / self._time_scale

        eq_out = self._integrate(t_norm)
        y_cart = self._geqoe_norm_to_cart(eq_out)

        stm = np.repeat(np.eye(6)[:, :, np.newaxis], N, axis=2)
        return y_cart, stm

    def resetInitialState(self, state: Any) -> None:
        """Reset the propagator to a new initial state."""
        self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
            _orekit_state_to_cartesian(state)
        )
        self._eq0 = self._cart_to_geqoe_norm(self._y0)

    def getInitialState(self) -> Any:
        """Return the current initial state as an Orekit ``SpacecraftState``."""
        if not self._orekit_available:
            raise RuntimeError("Orekit is not available.")
        return _orbit_to_state(self._initial_orbit, self._mass_kg)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def nfev(self) -> int:
        """Number of ODE RHS evaluations from the last integration."""
        return self._nfev

    @property
    def body_constants(self) -> dict[str, float]:
        return dict(self._body_constants)


# ---------------------------------------------------------------------------
# Orekit AbstractPropagator factory
# ---------------------------------------------------------------------------


def make_orekit_numerical_geqoe_propagator(
    initial_orbit: Any,
    body_constants: Mapping[str, float] | None = None,
    mass_kg: float = 1000.0,
    integrator: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> Any:
    """Create a ``NumericalGEqOEPropagator`` with Orekit ``AbstractPropagator`` base.

    Falls back to the plain Python class when Orekit is not available.
    """
    try:
        from org.orekit.propagation import AbstractPropagator as _AP  # noqa: F401

        if not hasattr(make_orekit_numerical_geqoe_propagator, "_cls"):

            class _OrekitNumericalGEqOEPropagator(_AP):  # type: ignore[misc]
                """NumericalGEqOEPropagator with Orekit AbstractPropagator base."""

                def __init__(
                    self, initial_orbit, body_constants, mass_kg,
                    integrator, rtol, atol,
                ):
                    _AP.__init__(self)
                    self._impl = NumericalGEqOEPropagator(
                        initial_orbit=initial_orbit,
                        body_constants=body_constants,
                        mass_kg=mass_kg,
                        integrator=integrator,
                        rtol=rtol,
                        atol=atol,
                    )
                    self.resetInitialState(self._impl.getInitialState())

                def propagateOrbit(self, date):
                    state = self._impl.propagate(date)
                    return state.getOrbit()

                def getMass(self, date):
                    return self._impl._mass_kg

                def resetIntermediateState(self, state, forward):
                    self._impl.resetInitialState(state)

                def propagate(self, *args):
                    return self._impl.propagate(*args)

                def propagate_array(self, dt_seconds):
                    return self._impl.propagate_array(dt_seconds)

                @property
                def nfev(self):
                    return self._impl.nfev

                @property
                def body_constants(self):
                    return self._impl.body_constants

            make_orekit_numerical_geqoe_propagator._cls = (  # type: ignore[attr-defined]
                _OrekitNumericalGEqOEPropagator
            )

        cls = make_orekit_numerical_geqoe_propagator._cls  # type: ignore[attr-defined]
        return cls(initial_orbit, body_constants, mass_kg, integrator, rtol, atol)

    except Exception:
        return NumericalGEqOEPropagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            mass_kg=mass_kg,
            integrator=integrator,
            rtol=rtol,
            atol=atol,
        )


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NumericalGEqOEProvider:
    """Provider for the numerical GEqOE propagator.

    Registered under ``kind="geqoe-numerical"``.
    """

    kind: str = "geqoe-numerical"
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=False,
        supports_propagator=True,
        supports_stm=False,
        is_analytical=False,
        supports_custom_output=True,
    )

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        body_constants = context.require_body_constants()
        initial_orbit = context.require_initial_orbit()
        integrator = spec.orekit_options.get("integrator", "DOP853")
        rtol = spec.orekit_options.get("rtol", 1e-12)
        atol = spec.orekit_options.get("atol", 1e-12)
        return make_orekit_numerical_geqoe_propagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            mass_kg=spec.mass_kg,
            integrator=integrator,
            rtol=rtol,
            atol=atol,
        )


__all__ = [
    "NumericalGEqOEPropagator",
    "NumericalGEqOEProvider",
    "make_orekit_numerical_geqoe_propagator",
]
