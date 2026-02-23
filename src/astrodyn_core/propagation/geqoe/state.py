from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np


ScratchValue = Union[float, int, bool, np.ndarray]


@dataclass(slots=True)
class GEqOEState:
    nu: float
    q1: float
    q2: float
    p1: float
    p2: float
    lr: float

    @classmethod
    def from_array(cls, values: np.ndarray) -> "GEqOEState":
        vec = np.asarray(values, dtype=float).reshape(6)
        return cls(*vec.tolist())

    def as_array(self) -> np.ndarray:
        return np.array([self.nu, self.q1, self.q2, self.p1, self.p2, self.lr], dtype=float)


@dataclass(slots=True)
class GEqOEPropagationConstants:
    j2: float
    re: float
    mu: float
    length_scale: float
    time_scale: float
    mu_norm: float
    a_half_j2: float


@dataclass(slots=True)
class GEqOEPropagationContext:
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
