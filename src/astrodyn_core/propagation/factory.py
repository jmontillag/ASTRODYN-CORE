"""High-level factory for builder/propagator construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.specs import PropagatorSpec


@dataclass(slots=True)
class PropagatorFactory:
    """Build builders or propagators from specs via a provider registry.

    Args:
        registry: Provider registry used to resolve builder/propagator lanes by
            ``spec.kind``.
    """

    registry: ProviderRegistry = field(default_factory=ProviderRegistry)

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a provider-specific builder for a propagator spec.

        Args:
            spec: Declarative propagator configuration.
            context: Runtime build context.

        Returns:
            Provider-specific builder object.
        """
        provider = self.registry.get_builder_provider(spec.kind)
        return provider.build_builder(spec, context)

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a provider-specific propagator for a propagator spec.

        Args:
            spec: Declarative propagator configuration.
            context: Runtime build context.

        Returns:
            Provider-specific propagator object.
        """
        provider = self.registry.get_propagator_provider(spec.kind)
        return provider.build_propagator(spec, context)
