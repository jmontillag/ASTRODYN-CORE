"""GEqOE analytical propagator — Orekit ``AbstractPropagator`` adapter.

Wraps the pure-Python Taylor-series J2 propagator
(:func:`~astrodyn_core.propagation.geqoe.core.taylor_cart_propagator`)
inside an Orekit ``AbstractPropagator`` subclass so that it participates
in all standard downstream workflows (trajectory export, STM extraction,
mission execution, etc.).

All Orekit imports are deferred to method bodies so that the module can
be imported without Orekit installed (useful for registry tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_log = logging.getLogger(__name__)

from astrodyn_core.propagation.geqoe.conversion import BodyConstants
from astrodyn_core.propagation.geqoe.core import evaluate_cart_taylor, prepare_cart_coefficients
from astrodyn_core.propagation.geqoe.state import GEqOETaylorCoefficients


# ---------------------------------------------------------------------------
# Helpers (Orekit-dependent, lazy-imported)
# ---------------------------------------------------------------------------


def _orekit_state_to_cartesian(state: Any) -> tuple[np.ndarray, Any, Any, float, float]:
    """Extract Cartesian position/velocity from an Orekit SpacecraftState.

    Returns
    -------
    y0 : ndarray, shape (6,)
        ``[rx, ry, rz, vx, vy, vz]`` in metres / metres-per-second.
    frame : Orekit Frame
    date : Orekit AbsoluteDate
    mu : float  (m^3/s^2)
    mass : float (kg)
    """
    orbit = state.getOrbit()
    frame = orbit.getFrame()
    pv = state.getPVCoordinates(frame)
    pos = pv.getPosition()
    vel = pv.getVelocity()
    y0 = np.array(
        [
            float(pos.getX()),
            float(pos.getY()),
            float(pos.getZ()),
            float(vel.getX()),
            float(vel.getY()),
            float(vel.getZ()),
        ],
        dtype=float,
    )
    mu = float(orbit.getMu())
    mass = float(state.getMass()) if hasattr(state, "getMass") else 1000.0
    return y0, frame, state.getDate(), mu, mass


def _cartesian_to_orekit_state(
    y: np.ndarray,
    frame: Any,
    date: Any,
    mu: float,
    mass: float,
) -> Any:
    """Build an Orekit ``SpacecraftState`` from a Cartesian state vector."""
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.orbits import CartesianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import PVCoordinates

    pos = Vector3D(float(y[0]), float(y[1]), float(y[2]))
    vel = Vector3D(float(y[3]), float(y[4]), float(y[5]))
    pv = PVCoordinates(pos, vel)
    orbit = CartesianOrbit(pv, frame, date, mu)
    return SpacecraftState(orbit, mass)


def _orbit_to_state(orbit: Any, mass: float = 1000.0) -> Any:
    """Build an Orekit ``SpacecraftState`` from an Orekit ``Orbit``."""
    from org.orekit.propagation import SpacecraftState

    return SpacecraftState(orbit, mass)


def _resolve_body_constants_from_orekit() -> dict[str, float]:
    """Resolve Earth body constants from Orekit ``Constants`` (WGS84).

    Mirrors the fallback logic in ``BuildContext.require_body_constants()``
    so that the direct adapter path behaves identically to the provider
    pipeline.
    """
    try:
        from org.orekit.utils import Constants
    except Exception as exc:
        raise RuntimeError(
            "body_constants not provided and Orekit is unavailable. "
            "Pass body_constants explicitly or ensure Orekit is initialized."
        ) from exc
    return {
        "mu": float(Constants.WGS84_EARTH_MU),
        "j2": float(-Constants.WGS84_EARTH_C20),
        "re": float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS),
    }


def _check_mu_consistency(orbit_mu: float, body_mu: float) -> None:
    """Warn if the orbit mu and body_constants mu disagree."""
    rel = abs(orbit_mu - body_mu) / max(abs(orbit_mu), 1.0)
    if rel > 1e-10:
        _log.warning(
            "mu from orbit (%.10e) differs from body_constants (%.10e) "
            "by %.2e relative. The J2 integration uses body_constants while "
            "the output Orekit orbit carries the orbit mu.",
            orbit_mu,
            body_mu,
            rel,
        )


# ---------------------------------------------------------------------------
# GEqOEPropagator  —  Orekit AbstractPropagator subclass
# ---------------------------------------------------------------------------


@dataclass
class _NativeResult:
    """Container for backend-specific GEqOE propagation output."""

    geqoe_cartesian: np.ndarray  # (6,) Cartesian at target epoch
    stm: np.ndarray  # (6, 6) state transition matrix


class GEqOEPropagator:
    """J2 Taylor-series propagator wrapped as an Orekit ``AbstractPropagator``.

    The class is designed so that its ``__init_subclass__`` / actual Orekit base
    class is resolved lazily: when Orekit is available, it subclasses
    ``AbstractPropagator`` at class-creation time via :func:`_make_propagator_class`.
    When Orekit is **not** available, it still works as a plain Python object
    that exposes the same public interface (minus Orekit-specific Java
    interop), which is sufficient for registry and unit tests that do not
    call ``propagate()``.

    Parameters
    ----------
    initial_orbit : Any
        Orekit ``Orbit`` object representing the initial state.
    body_constants : Mapping[str, float] | None
        Mapping with ``"mu"`` (m^3/s^2), ``"j2"`` (dimensionless),
        ``"re"`` (equatorial radius, m).  When *None* (the default), the
        constants are resolved from ``org.orekit.utils.Constants``
        (WGS84), which requires Orekit to be initialised.
    order : int
        Taylor expansion order (1–4).
    mass_kg : float
        Spacecraft mass in kilograms.
    """

    def __init__(
        self,
        initial_orbit: Any,
        body_constants: Mapping[str, float] | None = None,
        order: int = 4,
        mass_kg: float = 1000.0,
    ) -> None:
        # Try to initialise as a proper AbstractPropagator subclass.
        # If Orekit is not available, we skip the super().__init__() call
        # but still store all internal state so that non-propagate methods
        # (capabilities, get_native_state, etc.) remain functional.
        self._orekit_available = False
        try:
            from org.orekit.propagation import AbstractPropagator  # noqa: F401

            # Dynamically make this instance behave like an AbstractPropagator.
            # JPype subclassing requires class-level inheritance; we handle
            # that via the factory function below.  Here we just flag availability.
            self._orekit_available = True
        except Exception:
            pass

        # Resolve body constants from Orekit when not provided explicitly.
        if body_constants is None:
            body_constants = _resolve_body_constants_from_orekit()

        self._order = int(order)
        self._mass_kg = float(mass_kg)
        self._body_constants = dict(body_constants)
        self._bc = BodyConstants(
            j2=float(body_constants["j2"]),
            re=float(body_constants["re"]),
            mu=float(body_constants["mu"]),
        )
        self._initial_orbit = initial_orbit

        # Cache initial Cartesian state for propagation
        if self._orekit_available and initial_orbit is not None:
            initial_state = _orbit_to_state(initial_orbit, mass_kg)
            self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
                _orekit_state_to_cartesian(initial_state)
            )
            # Warn if the orbit's mu disagrees with body_constants["mu"].
            _check_mu_consistency(self._mu, self._bc.mu)
            # Precompute dt-independent Taylor coefficients and epoch Jacobian.
            # These are reused across all propagate() / propagate_array() calls
            # until the next resetInitialState().
            self._coeffs, self._peq_py_0 = prepare_cart_coefficients(
                self._y0, self._bc, self._order
            )
        else:
            self._y0 = None
            self._frame = None
            self._epoch = None
            self._mu = float(body_constants["mu"])
            self._coeffs: GEqOETaylorCoefficients | None = None
            self._peq_py_0: np.ndarray | None = None

        self._last_native: _NativeResult | None = None

    # ------------------------------------------------------------------
    # Orekit-compatible interface
    # ------------------------------------------------------------------

    def propagate(self, start: Any, target: Any | None = None) -> Any:
        """Propagate from *start* to *target* and return a ``SpacecraftState``.

        Follows the Orekit ``AbstractPropagator.propagate(start, target)``
        two-argument convention.  When called with a single argument, that
        argument is treated as the target date and the initial epoch is used
        as the start date.
        """
        if target is None:
            # Single-argument form: propagate(target_date)
            target = start
            start = self._epoch

        if not self._orekit_available:
            raise RuntimeError(
                "Orekit is not available. Cannot call propagate() without Orekit."
            )

        dt_seconds = float(target.durationFrom(self._epoch))
        tspan = np.array([dt_seconds], dtype=float)

        # Use the cached coefficients — evaluate_cart_taylor only does the
        # cheap polynomial evaluation + Cartesian conversion (no re-computation
        # of Taylor coefficients).
        assert self._coeffs is not None and self._peq_py_0 is not None, (
            "Taylor coefficients not initialised. Orekit must be available at construction."
        )
        y_out, stm = evaluate_cart_taylor(self._coeffs, self._peq_py_0, tspan)

        y_final = y_out[0, :]  # shape (6,)
        stm_final = stm[:, :, 0]  # shape (6, 6)

        self._last_native = _NativeResult(
            geqoe_cartesian=y_final,
            stm=stm_final,
        )

        return _cartesian_to_orekit_state(
            y_final,
            frame=self._frame,
            date=target,
            mu=self._mu,
            mass=self._mass_kg,
        )

    def resetInitialState(self, state: Any) -> None:
        """Reset the propagator to a new initial state.

        Required by the Orekit ``AbstractPropagator`` interface for mission
        simulation workflows that apply impulsive maneuvers or estimator
        iterations.  The Taylor coefficients are recomputed once from the
        new state so that subsequent ``propagate()`` calls remain fast.
        """
        self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
            _orekit_state_to_cartesian(state)
        )
        # Recompute the precomputed cache for the new initial state.
        self._coeffs, self._peq_py_0 = prepare_cart_coefficients(
            self._y0, self._bc, self._order
        )
        self._last_native = None

    def getInitialState(self) -> Any:
        """Return the current initial state as an Orekit ``SpacecraftState``."""
        if not self._orekit_available:
            raise RuntimeError("Orekit is not available.")
        return _orbit_to_state(self._initial_orbit, self._mass_kg)

    # ------------------------------------------------------------------
    # Backend-specific output
    # ------------------------------------------------------------------

    def get_native_state(self, target_date: Any) -> tuple[np.ndarray, np.ndarray]:
        """Propagate and return raw numpy Cartesian state + STM.

        Parameters
        ----------
        target_date : Orekit AbsoluteDate
            Target epoch.

        Returns
        -------
        y : ndarray, shape (6,)
            Cartesian state ``[rx, ry, rz, vx, vy, vz]`` (SI).
        stm : ndarray, shape (6, 6)
            State transition matrix mapping initial → final state.
        """
        # Propagate (populates self._last_native as a side-effect)
        self.propagate(target_date)
        assert self._last_native is not None
        return self._last_native.geqoe_cartesian.copy(), self._last_native.stm.copy()

    def propagate_array(
        self,
        dt_seconds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch propagation returning raw numpy arrays (no Orekit objects).

        Parameters
        ----------
        dt_seconds : ndarray, shape (N,)
            Time offsets from the initial epoch in seconds.

        Returns
        -------
        y_out : ndarray, shape (N, 6)
            Cartesian states at each time step.
        stm : ndarray, shape (6, 6, N)
            State transition matrices at each time step.
        """
        if self._coeffs is None or self._peq_py_0 is None:
            raise RuntimeError(
                "Taylor coefficients not available. "
                "Orekit may not have been initialised when the propagator was created."
            )
        tspan = np.atleast_1d(np.asarray(dt_seconds, dtype=float))
        return evaluate_cart_taylor(self._coeffs, self._peq_py_0, tspan)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def order(self) -> int:
        return self._order

    @property
    def body_constants(self) -> dict[str, float]:
        return dict(self._body_constants)


# ---------------------------------------------------------------------------
# JPype AbstractPropagator factory  (optional — used when Orekit is available)
# ---------------------------------------------------------------------------


def make_orekit_geqoe_propagator(
    initial_orbit: Any,
    body_constants: Mapping[str, float] | None = None,
    order: int = 4,
    mass_kg: float = 1000.0,
) -> Any:
    """Create a ``GEqOEPropagator`` that is also an Orekit ``AbstractPropagator``.

    When Orekit is available, this returns an instance of a dynamically
    created class that inherits from both ``GEqOEPropagator`` and
    ``AbstractPropagator``, giving full Java-side interoperability.

    When Orekit is **not** available, this falls back to a plain
    ``GEqOEPropagator`` instance.
    """
    try:
        from org.orekit.propagation import AbstractPropagator as _AP  # noqa: F401

        # JPype allows dynamic class creation.  We create a subclass once
        # and cache it on the module.
        if not hasattr(make_orekit_geqoe_propagator, "_cls"):

            class _OrekitGEqOEPropagator(_AP):  # type: ignore[misc]
                """GEqOEPropagator with Orekit AbstractPropagator base."""

                def __init__(self, initial_orbit, body_constants, order, mass_kg):
                    _AP.__init__(self)
                    # Re-use GEqOEPropagator init logic
                    self._impl = GEqOEPropagator(
                        initial_orbit=initial_orbit,
                        body_constants=body_constants,
                        order=order,
                        mass_kg=mass_kg,
                    )
                    # Set the initial state on the Java side
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

                def get_native_state(self, target_date):
                    return self._impl.get_native_state(target_date)

                def propagate_array(self, dt_seconds):
                    return self._impl.propagate_array(dt_seconds)

                @property
                def order(self):
                    return self._impl.order

                @property
                def body_constants(self):
                    return self._impl.body_constants

            make_orekit_geqoe_propagator._cls = _OrekitGEqOEPropagator  # type: ignore[attr-defined]

        cls = make_orekit_geqoe_propagator._cls  # type: ignore[attr-defined]
        return cls(initial_orbit, body_constants, order, mass_kg)

    except Exception:
        # Orekit not available — return plain Python propagator
        return GEqOEPropagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            order=order,
            mass_kg=mass_kg,
        )
