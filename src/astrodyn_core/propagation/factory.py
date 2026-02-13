"""High-level factory for builder/propagator construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.specs import PropagatorSpec


@dataclass(slots=True)
class PropagatorFactory:
    """Builds Orekit-native builders or propagators from specs."""

    registry: ProviderRegistry = field(default_factory=ProviderRegistry)

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        provider = self.registry.get_builder_provider(spec.kind)
        return provider.build_builder(spec, context)

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        provider = self.registry.get_propagator_provider(spec.kind)
        return provider.build_propagator(spec, context)
