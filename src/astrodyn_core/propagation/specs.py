"""Declarative propagation specifications."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.forces import ForceSpec
from astrodyn_core.propagation.spacecraft import SpacecraftSpec


class PropagatorKind(str, Enum):
    """Built-in propagator kinds.

    Custom/analytical propagators should define their own string kind
    (e.g. ``"geqoe"``) and register it with the :class:`ProviderRegistry`.
    The registry accepts any string as a kind key, so contributors are not
    required to modify this enum when adding new propagators.
    """

    NUMERICAL = "numerical"
    KEPLERIAN = "keplerian"
    DSST = "dsst"
    TLE = "tle"


@dataclass(frozen=True, slots=True)
class IntegratorSpec:
    """Integrator builder configuration for numerical and DSST builders.

    Attributes:
        kind: Integrator kind identifier (for example ``"dp853"``).
        min_step: Minimum integration step size in seconds (adaptive methods).
        max_step: Maximum integration step size in seconds (adaptive methods).
        position_tolerance: Position tolerance used to derive Orekit tolerances.
        step: Fixed step size in seconds (fixed-step methods).
        n_steps: Optional auxiliary step count for custom providers.
        extra: Provider-specific integrator options.
    """

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
    """Raw TLE line pair for SGP4/TLE propagator construction.

    Attributes:
        line1: TLE line 1 text (must start with ``"1 "``).
        line2: TLE line 2 text (must start with ``"2 "``).
    """

    line1: str
    line2: str

    def __post_init__(self) -> None:
        if not self.line1.startswith("1 "):
            raise ValueError("TLE line1 must start with '1 '.")
        if not self.line2.startswith("2 "):
            raise ValueError("TLE line2 must start with '2 '.")


@dataclass(frozen=True, slots=True)
class PropagatorSpec:
    """Top-level propagation configuration.

    The ``kind`` field accepts any :class:`PropagatorKind` enum member for
    built-in Orekit propagators, or a plain string for custom/analytical
    propagators registered via :class:`ProviderRegistry`.

    Attributes:
        kind: Built-in or custom propagator kind key.
        mass_kg: Spacecraft mass used by builders/propagators.
        position_angle_type: Orekit position-angle convention for relevant
            propagator builders.
        dsst_propagation_type: DSST propagation type (``MEAN``/``OSCULATING``).
        dsst_state_type: DSST initial state type (``MEAN``/``OSCULATING``).
        integrator: Integrator configuration (required for numerical/DSST).
        tle: Raw TLE lines (required for TLE propagator).
        force_specs: Declarative force model specs.
        spacecraft: Spacecraft physical model for drag/SRP assembly.
        attitude: Declarative attitude configuration.
        orekit_options: Provider-specific Orekit/native options passthrough.
    """

    kind: PropagatorKind | str
    mass_kg: float = 1000.0
    position_angle_type: str = "MEAN"
    dsst_propagation_type: str = "MEAN"
    dsst_state_type: str = "OSCULATING"
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
        dsst_propagation_type = self.dsst_propagation_type.strip().upper()
        dsst_state_type = self.dsst_state_type.strip().upper()
        object.__setattr__(self, "dsst_propagation_type", dsst_propagation_type)
        object.__setattr__(self, "dsst_state_type", dsst_state_type)

        # Normalize kind to its string value for comparison.
        # Works for both PropagatorKind enum members and plain strings.
        kind_val = self.kind.value if isinstance(self.kind, PropagatorKind) else self.kind
        if kind_val in (PropagatorKind.NUMERICAL.value, PropagatorKind.DSST.value):
            if self.integrator is None:
                raise ValueError(f"integrator is required for kind={kind_val!r}.")

        if kind_val == PropagatorKind.TLE.value and self.tle is None:
            raise ValueError("tle is required for kind=tle.")

        if kind_val == PropagatorKind.DSST.value:
            valid = {"MEAN", "OSCULATING"}
            if dsst_propagation_type not in valid:
                raise ValueError(
                    "dsst_propagation_type must be one of {'MEAN', 'OSCULATING'} for kind=dsst."
                )
            if dsst_state_type not in valid:
                raise ValueError(
                    "dsst_state_type must be one of {'MEAN', 'OSCULATING'} for kind=dsst."
                )

    def with_spacecraft(self, spacecraft: SpacecraftSpec) -> PropagatorSpec:
        """Return a copy of the spec with a spacecraft model attached.

        Args:
            spacecraft: Spacecraft physical model specification.

        Returns:
            New ``PropagatorSpec`` with ``spacecraft`` replaced.
        """
        return replace(self, spacecraft=spacecraft)
