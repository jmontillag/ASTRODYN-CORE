"""Adaptive checkpoint-based GEqOE Taylor propagator.

Chains multiple Taylor expansions end-to-end so that the truncation error
stays below a user-specified position tolerance.  Checkpoints are computed
lazily: only when a query time falls outside the range of all existing
checkpoints.

The resulting propagator behaves like a general-purpose J2 analytical
propagator with controlled accuracy and orders-of-magnitude speed
advantage over numerical integration for long-duration arcs.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.geqoe.conversion import BodyConstants
from astrodyn_core.propagation.geqoe.error import compute_max_dt
from astrodyn_core.propagation.geqoe.jacobians import get_pYpEq
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
# Checkpoint data structure
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Checkpoint:
    """A single Taylor expansion segment along the propagation arc.

    Attributes:
        epoch_seconds: Absolute time offset from the initial epoch (seconds).
        y0_cart: Cartesian state ``(6,)`` at this checkpoint.
        coeffs: Taylor coefficients (``GEqOETaylorCoefficients`` or C++ capsule).
        peq_py_0: ``(6, 6)`` epoch Jacobian d(GEqOE)/d(Cart) at this checkpoint.
        dt_max: Maximum ``|dt|`` from this checkpoint (seconds).
        cumulative_stm: ``(6, 6)`` STM from ``t=0`` to this checkpoint epoch.
    """

    epoch_seconds: float
    y0_cart: np.ndarray
    coeffs: Any
    peq_py_0: np.ndarray
    dt_max: float
    cumulative_stm: np.ndarray


# ---------------------------------------------------------------------------
# Adaptive GEqOE propagator
# ---------------------------------------------------------------------------


class AdaptiveGEqOEPropagator:
    """J2 Taylor-series propagator with adaptive checkpoint chaining.

    Parameters
    ----------
    initial_orbit : Any
        Orekit ``Orbit`` object representing the initial state.
    body_constants : Mapping[str, float] | None
        ``{"mu", "j2", "re"}`` in SI.  Resolved from Orekit WGS84 when *None*.
    order : int
        Taylor expansion order (1-4).
    mass_kg : float
        Spacecraft mass in kilograms.
    backend : str
        ``"cpp"`` (default) or ``"python"``.
    pos_tol : float
        Position tolerance in meters per checkpoint step.
    safety_factor : float
        Multiplicative safety margin on ``dt_max`` (default 0.8).
    max_step : float | None
        When set, caps ``dt_max`` at this value (seconds).  Overrides the
        error-based step controller — useful for fixed-step diagnostics.
    """

    def __init__(
        self,
        initial_orbit: Any,
        body_constants: Mapping[str, float] | None = None,
        order: int = 4,
        mass_kg: float = 1000.0,
        backend: str = "cpp",
        pos_tol: float = 1.0,
        safety_factor: float = 0.8,
        max_step: float | None = None,
    ) -> None:
        self._orekit_available = False
        try:
            from org.orekit.propagation import AbstractPropagator  # noqa: F401

            self._orekit_available = True
        except Exception:
            pass

        if body_constants is None:
            body_constants = _resolve_body_constants_from_orekit()

        if backend not in ("cpp", "python"):
            raise ValueError(f"backend must be 'cpp' or 'python', got {backend!r}")

        self._backend = backend
        self._order = int(order)
        self._mass_kg = float(mass_kg)
        self._body_constants = dict(body_constants)
        self._bc = BodyConstants(
            j2=float(body_constants["j2"]),
            re=float(body_constants["re"]),
            mu=float(body_constants["mu"]),
        )
        self._pos_tol = float(pos_tol)
        self._safety_factor = float(safety_factor)
        self._max_step = float(max_step) if max_step is not None else None
        self._time_scale = float((self._bc.re**3 / self._bc.mu) ** 0.5)
        self._initial_orbit = initial_orbit

        # Sorted checkpoint list (by epoch_seconds).
        self._checkpoints: list[Checkpoint] = []

        if self._orekit_available and initial_orbit is not None:
            initial_state = _orbit_to_state(initial_orbit, mass_kg)
            self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
                _orekit_state_to_cartesian(initial_state)
            )
            _check_mu_consistency(self._mu, self._bc.mu)
            self._build_initial_checkpoint()
        else:
            self._y0 = None
            self._frame = None
            self._epoch = None
            self._mu = float(body_constants["mu"])

    # ------------------------------------------------------------------
    # Backend dispatch helpers (mirrors GEqOEPropagator)
    # ------------------------------------------------------------------

    def _prepare(self, y0: np.ndarray) -> tuple[Any, np.ndarray]:
        """Prepare Taylor coefficients using the configured backend."""
        if self._backend == "cpp":
            from astrodyn_core.geqoe_cpp import prepare_cart_coefficients_cpp

            return prepare_cart_coefficients_cpp(
                y0, self._bc.j2, self._bc.re, self._bc.mu, self._order
            )
        from astrodyn_core.propagation.geqoe.core import prepare_cart_coefficients

        return prepare_cart_coefficients(y0, self._bc, self._order)

    def _evaluate_at(
        self, coeffs: Any, peq_py_0: np.ndarray, tspan: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate Taylor expansion at *tspan* offsets from the expansion epoch."""
        if self._backend == "cpp":
            from astrodyn_core.geqoe_cpp import evaluate_cart_taylor_cpp

            return evaluate_cart_taylor_cpp(coeffs, peq_py_0, tspan)
        from astrodyn_core.propagation.geqoe.core import evaluate_cart_taylor

        return evaluate_cart_taylor(coeffs, peq_py_0, tspan)

    def _extract_map_components(self, coeffs: Any) -> np.ndarray:
        """Extract ``(6, order)`` map_components from coefficients."""
        if self._backend == "cpp":
            from astrodyn_core.geqoe_cpp import evaluate_taylor_cpp

            _, _, map_comps = evaluate_taylor_cpp(coeffs, np.array([0.0]))
            return np.asarray(map_comps)
        return coeffs.map_components.copy()

    def _extract_geqoe_state(self, coeffs: Any) -> np.ndarray:
        """Extract the GEqOE state at the expansion epoch."""
        if self._backend == "cpp":
            from astrodyn_core.geqoe_cpp import evaluate_taylor_cpp

            y_prop, _, _ = evaluate_taylor_cpp(coeffs, np.array([0.0]))
            return np.asarray(y_prop[0])
        return coeffs.initial_geqoe.copy()

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _make_checkpoint(
        self,
        epoch_seconds: float,
        y0_cart: np.ndarray,
        cumulative_stm: np.ndarray,
    ) -> Checkpoint:
        """Build a checkpoint at the given Cartesian state."""
        coeffs, peq_py_0 = self._prepare(y0_cart)

        if self._max_step is not None:
            dt_max = self._max_step
        else:
            map_comps = self._extract_map_components(coeffs)
            geqoe_state = self._extract_geqoe_state(coeffs)

            pYpEq = get_pYpEq(
                t=0.0,
                y=geqoe_state,
                p=(self._bc.j2, self._bc.re, self._bc.mu),
            )
            if pYpEq.ndim == 3:
                pYpEq = pYpEq[0]

            dt_max = compute_max_dt(
                map_components=map_comps,
                pYpEq_epoch=pYpEq,
                time_scale=self._time_scale,
                order=self._order,
                pos_tol=self._pos_tol,
                safety_factor=self._safety_factor,
            )

        return Checkpoint(
            epoch_seconds=epoch_seconds,
            y0_cart=y0_cart.copy(),
            coeffs=coeffs,
            peq_py_0=peq_py_0,
            dt_max=dt_max,
            cumulative_stm=cumulative_stm.copy(),
        )

    def _build_initial_checkpoint(self) -> None:
        """Create the initial checkpoint at t=0."""
        assert self._y0 is not None
        ckpt = self._make_checkpoint(
            epoch_seconds=0.0,
            y0_cart=self._y0,
            cumulative_stm=np.eye(6),
        )
        self._checkpoints = [ckpt]

    def _find_or_extend(self, t: float) -> Checkpoint:
        """Find (or create) the checkpoint that covers query time *t*.

        Searches existing checkpoints for one where ``|t - epoch| <= dt_max``.
        If none is found, extends the checkpoint chain toward *t*.
        """
        # Search existing checkpoints for the best candidate.
        epochs = [c.epoch_seconds for c in self._checkpoints]
        idx = bisect.bisect_left(epochs, t)

        # Check the two nearest checkpoints (left and right of insertion point).
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self._checkpoints):
            candidates.append(idx)

        best_i = None
        best_dist = np.inf
        for ci in candidates:
            dist = abs(t - self._checkpoints[ci].epoch_seconds)
            if dist <= self._checkpoints[ci].dt_max and dist < best_dist:
                best_i = ci
                best_dist = dist

        if best_i is not None:
            return self._checkpoints[best_i]

        # Need to extend — determine direction.
        if t >= 0:
            # Extend forward from last checkpoint.
            boundary = self._checkpoints[-1]
        else:
            # Extend backward from first checkpoint.
            boundary = self._checkpoints[0]

        return self._extend_toward(boundary, t)

    def _extend_toward(self, boundary: Checkpoint, t: float) -> Checkpoint:
        """Extend the checkpoint chain from *boundary* toward time *t*."""
        current = boundary
        while True:
            direction = 1.0 if t >= current.epoch_seconds else -1.0
            step = direction * current.dt_max
            new_epoch = current.epoch_seconds + step

            # Evaluate Taylor at the step boundary to get the new Cartesian state.
            y_out, stm = self._evaluate_at(
                current.coeffs, current.peq_py_0, np.array([step])
            )
            new_y0 = y_out[0]
            local_stm = stm[:, :, 0]  # (6, 6)
            new_cumulative = local_stm @ current.cumulative_stm

            new_ckpt = self._make_checkpoint(new_epoch, new_y0, new_cumulative)

            # Insert in sorted order.
            epochs = [c.epoch_seconds for c in self._checkpoints]
            insert_idx = bisect.bisect_left(epochs, new_epoch)
            self._checkpoints.insert(insert_idx, new_ckpt)

            # Check if the new checkpoint covers the query time.
            if abs(t - new_epoch) <= new_ckpt.dt_max:
                return new_ckpt

            current = new_ckpt

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

        dt = float(target.durationFrom(self._epoch))
        ckpt = self._find_or_extend(dt)
        local_dt = dt - ckpt.epoch_seconds

        y_out, local_stm = self._evaluate_at(
            ckpt.coeffs, ckpt.peq_py_0, np.array([local_dt])
        )
        return _cartesian_to_orekit_state(
            y_out[0],
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
            Tuple ``(y_out, stm)`` with Cartesian states ``(N, 6)`` and STMs
            ``(6, 6, N)``.
        """
        tspan = np.atleast_1d(np.asarray(dt_seconds, dtype=float))
        N = len(tspan)
        y_out = np.zeros((N, 6))
        stm_out = np.zeros((6, 6, N))

        # Group queries by checkpoint for batch evaluation.
        groups: dict[int, list[int]] = {}  # ckpt_id → [query indices]
        ckpt_map: dict[int, Checkpoint] = {}

        for i, t in enumerate(tspan):
            ckpt = self._find_or_extend(t)
            ckpt_id = id(ckpt)
            if ckpt_id not in groups:
                groups[ckpt_id] = []
                ckpt_map[ckpt_id] = ckpt
            groups[ckpt_id].append(i)

        for ckpt_id, indices in groups.items():
            ckpt = ckpt_map[ckpt_id]
            local_dts = np.array(
                [tspan[i] - ckpt.epoch_seconds for i in indices], dtype=float
            )
            y_batch, stm_batch = self._evaluate_at(
                ckpt.coeffs, ckpt.peq_py_0, local_dts
            )
            for j, qi in enumerate(indices):
                y_out[qi] = y_batch[j]
                local_stm_j = stm_batch[:, :, j]
                stm_out[:, :, qi] = local_stm_j @ ckpt.cumulative_stm

        return y_out, stm_out

    def resetInitialState(self, state: Any) -> None:
        """Reset the propagator to a new initial state."""
        self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
            _orekit_state_to_cartesian(state)
        )
        self._build_initial_checkpoint()

    def getInitialState(self) -> Any:
        """Return the current initial state as an Orekit ``SpacecraftState``."""
        if not self._orekit_available:
            raise RuntimeError("Orekit is not available.")
        return _orbit_to_state(self._initial_orbit, self._mass_kg)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def order(self) -> int:
        return self._order

    @property
    def body_constants(self) -> dict[str, float]:
        return dict(self._body_constants)

    @property
    def num_checkpoints(self) -> int:
        return len(self._checkpoints)


# ---------------------------------------------------------------------------
# Orekit AbstractPropagator factory
# ---------------------------------------------------------------------------


def make_orekit_adaptive_geqoe_propagator(
    initial_orbit: Any,
    body_constants: Mapping[str, float] | None = None,
    order: int = 4,
    mass_kg: float = 1000.0,
    backend: str = "cpp",
    pos_tol: float = 1.0,
    safety_factor: float = 0.8,
    max_step: float | None = None,
) -> Any:
    """Create an ``AdaptiveGEqOEPropagator`` with Orekit ``AbstractPropagator`` base.

    Falls back to the plain Python class when Orekit is not available.
    """
    try:
        from org.orekit.propagation import AbstractPropagator as _AP  # noqa: F401

        if not hasattr(make_orekit_adaptive_geqoe_propagator, "_cls"):

            class _OrekitAdaptiveGEqOEPropagator(_AP):  # type: ignore[misc]
                """AdaptiveGEqOEPropagator with Orekit AbstractPropagator base."""

                def __init__(
                    self, initial_orbit, body_constants, order, mass_kg, backend,
                    pos_tol, safety_factor, max_step,
                ):
                    _AP.__init__(self)
                    self._impl = AdaptiveGEqOEPropagator(
                        initial_orbit=initial_orbit,
                        body_constants=body_constants,
                        order=order,
                        mass_kg=mass_kg,
                        backend=backend,
                        pos_tol=pos_tol,
                        safety_factor=safety_factor,
                        max_step=max_step,
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
                def order(self):
                    return self._impl.order

                @property
                def body_constants(self):
                    return self._impl.body_constants

                @property
                def num_checkpoints(self):
                    return self._impl.num_checkpoints

            make_orekit_adaptive_geqoe_propagator._cls = (  # type: ignore[attr-defined]
                _OrekitAdaptiveGEqOEPropagator
            )

        cls = make_orekit_adaptive_geqoe_propagator._cls  # type: ignore[attr-defined]
        return cls(
            initial_orbit, body_constants, order, mass_kg, backend,
            pos_tol, safety_factor, max_step,
        )

    except Exception:
        return AdaptiveGEqOEPropagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            order=order,
            mass_kg=mass_kg,
            backend=backend,
            pos_tol=pos_tol,
            safety_factor=safety_factor,
            max_step=max_step,
        )


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptiveGEqOEProvider:
    """Provider for the adaptive checkpoint-based GEqOE Taylor propagator.

    Registered under ``kind="geqoe-adaptive"``.
    """

    kind: str = "geqoe-adaptive"
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=False,
        supports_propagator=True,
        supports_stm=True,
        is_analytical=True,
        supports_custom_output=True,
    )

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        body_constants = context.require_body_constants()
        initial_orbit = context.require_initial_orbit()
        order = spec.orekit_options.get("taylor_order", 4)
        backend = spec.orekit_options.get("backend", "cpp")
        pos_tol = spec.orekit_options.get("pos_tol", 1.0)
        safety_factor = spec.orekit_options.get("safety_factor", 0.8)
        max_step = spec.orekit_options.get("max_step", None)
        return make_orekit_adaptive_geqoe_propagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            order=order,
            mass_kg=spec.mass_kg,
            backend=backend,
            pos_tol=pos_tol,
            safety_factor=safety_factor,
            max_step=max_step,
        )
