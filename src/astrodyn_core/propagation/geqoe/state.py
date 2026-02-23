from dataclasses import dataclass, field
from typing import Union

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
class GEqOEHistoryBuffers:
    beta: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    c: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    r: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    r2: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    r3: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    h: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    beta_plus_one: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    hr3: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    delta: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    delta_denominator: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))


@dataclass(slots=True)
class GEqOEPropagationContext:
    dt_seconds: np.ndarray
    dt_norm: np.ndarray
    initial_state: GEqOEState
    order: int
    constants: GEqOEPropagationConstants
    histories: GEqOEHistoryBuffers = field(default_factory=GEqOEHistoryBuffers)
    scratch: dict[str, ScratchValue] = field(default_factory=dict)
    y_prop: np.ndarray | None = None
    y_y0: np.ndarray | None = None
    map_components: np.ndarray | None = None
