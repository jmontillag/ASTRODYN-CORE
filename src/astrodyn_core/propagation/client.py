"""High-level propagation façade for common builder/propagator workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.propagation.parsers.dynamics import (
    load_dynamics_config,
    load_dynamics_from_dict,
)
from astrodyn_core.propagation.parsers.spacecraft import (
    load_spacecraft_config,
    load_spacecraft_from_dict,
)
from astrodyn_core.propagation.factory import PropagatorFactory
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.specs import PropagatorSpec
from astrodyn_core.propagation.providers import (
    register_all_providers,
    register_default_orekit_providers,
)
from astrodyn_core.states.models import OrbitStateRecord


@dataclass(slots=True)
class PropagationClient:
    """Facade for ergonomic propagation factory/context usage.

    Args:
        universe: Optional default universe configuration used when converting
            state records into Orekit orbits for build contexts.
    """

    universe: Mapping[str, Any] | None = None

    def build_factory(self) -> PropagatorFactory:
        """Create a PropagatorFactory with all built-in providers registered.

        This includes both Orekit-native providers (numerical, keplerian, DSST,
        TLE) and built-in analytical providers (currently ``geqoe``).

        Returns:
            A factory with built-in providers pre-registered.
        """
        registry = ProviderRegistry()
        register_all_providers(registry)
        return PropagatorFactory(registry=registry)

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a propagator builder from a spec and build context.

        Args:
            spec: Declarative propagator configuration.
            context: Runtime build context (initial orbit, forces, etc.).

        Returns:
            Provider-specific propagator builder (typically Orekit-native).
        """
        return self.build_factory().build_builder(spec, context)

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a propagator directly from a spec and build context.

        Args:
            spec: Declarative propagator configuration.
            context: Runtime build context (initial orbit, forces, etc.).

        Returns:
            Provider-specific propagator instance.
        """
        return self.build_factory().build_propagator(spec, context)

    def context_from_state(
        self,
        state: OrbitStateRecord,
        *,
        universe: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BuildContext:
        """Create a ``BuildContext`` from an ``OrbitStateRecord``.

        Args:
            state: Serializable initial orbit state.
            universe: Optional per-call universe config override.
            metadata: Optional metadata merged into the build context.

        Returns:
            Build context containing an Orekit orbit converted from ``state``.
        """
        selected_universe = universe if universe is not None else self.universe
        return BuildContext.from_state_record(state, universe=selected_universe, metadata=metadata)

    def build_propagator_from_state(
        self,
        state: OrbitStateRecord,
        spec: PropagatorSpec,
        *,
        universe: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Build a propagator from an initial state record and a spec.

        Works with any registered propagator kind (numerical, keplerian, DSST,
        TLE, or custom/analytical).

        Args:
            state: Serializable initial orbit state.
            spec: Declarative propagator configuration.
            universe: Optional per-call universe config override.
            metadata: Optional build-context metadata.

        Returns:
            Built propagator instance created from the provider's builder lane.
        """
        context = self.context_from_state(state, universe=universe, metadata=metadata)
        builder = self.build_builder(spec, context)
        return builder.buildPropagator(builder.getSelectedNormalizedParameters())

    def load_dynamics_config(
        self, path: str | Path, spacecraft: str | Path | None = None
    ) -> PropagatorSpec:
        """Load a dynamics YAML config as ``PropagatorSpec``.

        Args:
            path: Dynamics YAML file path.
            spacecraft: Optional spacecraft YAML file path to merge into the
                returned spec.

        Returns:
            Parsed propagator spec.
        """
        return load_dynamics_config(path, spacecraft=spacecraft)

    def load_dynamics_from_dict(self, data: dict[str, Any]) -> PropagatorSpec:
        """Load a dynamics config mapping as ``PropagatorSpec``.

        Args:
            data: Parsed dynamics configuration mapping.

        Returns:
            Parsed propagator spec.
        """
        return load_dynamics_from_dict(data)

    def load_spacecraft_config(self, path: str | Path):
        """Load a spacecraft YAML config as ``SpacecraftSpec``."""
        return load_spacecraft_config(path)

    def load_spacecraft_from_dict(self, data: dict[str, Any]):
        """Load a spacecraft config mapping as ``SpacecraftSpec``."""
        return load_spacecraft_from_dict(data)
