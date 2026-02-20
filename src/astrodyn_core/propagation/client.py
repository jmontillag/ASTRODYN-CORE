"""High-level propagation façade for common builder/propagator workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.propagation.config import (
    load_dynamics_config,
    load_dynamics_from_dict,
    load_spacecraft_config,
    load_spacecraft_from_dict,
)
from astrodyn_core.propagation.factory import PropagatorFactory
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.specs import PropagatorSpec
from astrodyn_core.propagation.providers.orekit_native import register_default_orekit_providers
from astrodyn_core.states.models import OrbitStateRecord


@dataclass(slots=True)
class PropagationClient:
    """Facade for ergonomic propagation factory/context usage."""

    universe: Mapping[str, Any] | None = None

    def build_factory(self) -> PropagatorFactory:
        """Create a PropagatorFactory with default Orekit providers registered."""
        registry = ProviderRegistry()
        register_default_orekit_providers(registry)
        return PropagatorFactory(registry=registry)

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a propagator builder from spec and context."""
        return self.build_factory().build_builder(spec, context)

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a propagator from spec and context."""
        return self.build_factory().build_propagator(spec, context)

    def context_from_state(
        self,
        state: OrbitStateRecord,
        *,
        universe: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BuildContext:
        """Create BuildContext from an OrbitStateRecord."""
        selected_universe = universe if universe is not None else self.universe
        return BuildContext.from_state_record(state, universe=selected_universe, metadata=metadata)

    def build_numerical_propagator_from_state(
        self,
        state: OrbitStateRecord,
        spec: PropagatorSpec,
        *,
        universe: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Build a propagator from an initial state record and a spec."""
        context = self.context_from_state(state, universe=universe, metadata=metadata)
        builder = self.build_builder(spec, context)
        return builder.buildPropagator(builder.getSelectedNormalizedParameters())

    def load_dynamics_config(self, path: str | Path, spacecraft: str | Path | None = None) -> PropagatorSpec:
        """Load dynamics YAML config as PropagatorSpec."""
        return load_dynamics_config(path, spacecraft=spacecraft)

    def load_dynamics_from_dict(self, data: dict[str, Any]) -> PropagatorSpec:
        """Load dynamics config from dict as PropagatorSpec."""
        return load_dynamics_from_dict(data)

    def load_spacecraft_config(self, path: str | Path):
        """Load spacecraft YAML config as SpacecraftSpec."""
        return load_spacecraft_config(path)

    def load_spacecraft_from_dict(self, data: dict[str, Any]):
        """Load spacecraft config from dict as SpacecraftSpec."""
        return load_spacecraft_from_dict(data)
