"""Declarative propagation specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.forces import ForceSpec
from astrodyn_core.propagation.spacecraft import SpacecraftSpec


class PropagatorKind(str, Enum):
    NUMERICAL = "numerical"
    KEPLERIAN = "keplerian"
    DSST = "dsst"
    TLE = "tle"


@dataclass(frozen=True, slots=True)
class IntegratorSpec:
    """Integrator builder configuration for numerical and DSST builders."""

    kind: str
    min_step: float | None = None
    max_step: float | None = None
    position_tolerance: float | None = None
    step: float | None = None
    n_steps: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        object.__setattr__(self, "kind", kind)
        if not kind:
            raise ValueError("IntegratorSpec.kind cannot be empty.")


@dataclass(frozen=True, slots=True)
class TLESpec:
    """Raw TLE line pair for SGP4-style construction."""

    line1: str
    line2: str

    def __post_init__(self) -> None:
        if not self.line1.startswith("1 "):
            raise ValueError("TLE line1 must start with '1 '.")
        if not self.line2.startswith("2 "):
            raise ValueError("TLE line2 must start with '2 '.")


@dataclass(frozen=True, slots=True)
class PropagatorSpec:
    """Top-level propagation configuration."""

    kind: PropagatorKind
    mass_kg: float = 1000.0
    position_angle_type: str = "MEAN"
    integrator: IntegratorSpec | None = None
    tle: TLESpec | None = None
    force_specs: Sequence[ForceSpec] = field(default_factory=tuple)
    spacecraft: SpacecraftSpec | None = None
    attitude: AttitudeSpec | None = None
    orekit_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive.")

        position_angle = self.position_angle_type.strip().upper()
        object.__setattr__(self, "position_angle_type", position_angle)

        if self.kind in (PropagatorKind.NUMERICAL, PropagatorKind.DSST):
            if self.integrator is None:
                raise ValueError(f"integrator is required for kind={self.kind.value}.")

        if self.kind == PropagatorKind.TLE and self.tle is None:
            raise ValueError("tle is required for kind=tle.")
