"""Core interfaces and context objects for propagation providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.specs import PropagatorKind, PropagatorSpec


@dataclass(slots=True)
class BuildContext:
    """Runtime context required by provider implementations."""

    initial_orbit: Any | None = None
    position_tolerance: float = 10.0
    attitude_provider: Any | None = None
    force_models: Sequence[Any] = field(default_factory=tuple)
    universe: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_initial_orbit(self) -> Any:
        if self.initial_orbit is None:
            raise ValueError("initial_orbit is required for this propagation kind.")
        return self.initial_orbit


class BuilderProvider(Protocol):
    """Provider that can build an Orekit PropagatorBuilder."""

    kind: PropagatorKind
    capabilities: CapabilityDescriptor

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Return an Orekit PropagatorBuilder-compatible instance."""


class PropagatorProvider(Protocol):
    """Provider that can create an Orekit Propagator directly."""

    kind: PropagatorKind
    capabilities: CapabilityDescriptor

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Return an Orekit Propagator-compatible instance."""
