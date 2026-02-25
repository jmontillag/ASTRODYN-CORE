"""GEqOE propagation state/context dataclasses used by staged backends."""

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np


ScratchValue = Union[float, int, bool, np.ndarray]


@dataclass(slots=True)
class GEqOEState:
    """GEqOE state vector decomposed into named fields.

    Attributes:
        nu: Generalized mean motion-like state component.
        q1: Inclination-related equinoctial component.
        q2: Inclination-related equinoctial component.
        p1: Eccentricity-related equinoctial component.
        p2: Eccentricity-related equinoctial component.
        lr: Generalized longitude variable.
    """

    nu: float
    q1: float
    q2: float
    p1: float
    p2: float
    lr: float

    @classmethod
    def from_array(cls, values: np.ndarray) -> "GEqOEState":
        """Create a state from a 6-element array-like vector."""
        vec = np.asarray(values, dtype=float).reshape(6)
        return cls(*vec.tolist())

    def as_array(self) -> np.ndarray:
        """Return the state as a NumPy vector in canonical field order."""
        return np.array([self.nu, self.q1, self.q2, self.p1, self.p2, self.lr], dtype=float)


@dataclass(slots=True)
class GEqOEPropagationConstants:
    """Normalized and physical constants used during staged propagation.

    Attributes:
        j2: J2 coefficient (dimensionless).
        re: Equatorial radius in meters.
        mu: Gravitational parameter in ``m^3/s^2``.
        length_scale: Normalization length scale (typically ``re``).
        time_scale: Normalization time scale ``sqrt(re^3/mu)``.
        mu_norm: Normalized gravitational parameter (typically ``1``).
        a_half_j2: Cached ``j2 / 2`` coefficient used by staged formulas.
    """

    j2: float
    re: float
    mu: float
    length_scale: float
    time_scale: float
    mu_norm: float
    a_half_j2: float


@dataclass(slots=True)
class GEqOEPropagationContext:
    """Mutable runtime context shared by staged Taylor-order functions.

    Attributes:
        dt_seconds: Requested time offsets in seconds.
        dt_norm: Normalized time offsets.
        initial_state: Initial GEqOE state.
        order: Taylor expansion order (1-4).
        constants: Normalized propagation constants.
        scratch: Shared scratch map for staged intermediate values.
        y_prop: Propagated GEqOE states (allocated/populated by staged code).
        y_y0: GEqOE-to-GEqOE STM tensor (allocated/populated by staged code).
        map_components: Taylor coefficient map components.
    """

    dt_seconds: np.ndarray
    dt_norm: np.ndarray
    initial_state: GEqOEState
    order: int
    constants: GEqOEPropagationConstants
    scratch: dict[str, ScratchValue] = field(default_factory=dict)
    y_prop: np.ndarray | None = None
    y_y0: np.ndarray | None = None
    map_components: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class GEqOETaylorCoefficients:
    """Precomputed Taylor coefficients (dt-independent).

    Holds everything needed to evaluate the Taylor polynomial at arbitrary
    time offsets without repeating the expensive coefficient computation.
    Created by :func:`~astrodyn_core.propagation.geqoe.core.prepare_taylor_coefficients`
    and consumed by :func:`~astrodyn_core.propagation.geqoe.core.evaluate_taylor`.
    """

    initial_geqoe: np.ndarray
    """(6,) GEqOE state at epoch."""

    peq_py_0: np.ndarray
    """(6, 6) Jacobian d(GEqOE)/d(Cartesian) at epoch."""

    constants: GEqOEPropagationConstants
    """Normalised propagation constants."""

    order: int
    """Taylor expansion order (1-4)."""

    scratch: dict[str, ScratchValue]
    """dt-independent scratch entries (Taylor coefficients + partials)."""

    map_components: np.ndarray
    """(6, order) Taylor coefficient matrix per order."""

    initial_state: GEqOEState
    """GEqOE state decomposed into named fields."""

    body_params: Any
    """BodyConstants or (j2, re, mu) tuple — needed for geqoe2rv/get_pYpEq."""
