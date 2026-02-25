"""Default Orekit-native provider implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrodyn_core.propagation.assembly_parts import assemble_attitude_provider, assemble_force_models
from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models
from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.orekit_env import get_mu
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.providers.integrators import create_orekit_integrator_builder
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.spacecraft import SpacecraftSpec
from astrodyn_core.propagation.specs import PropagatorKind, PropagatorSpec


def _resolve_attitude(spec: PropagatorSpec, context: BuildContext, orbit: Any) -> Any | None:
    """Resolve an attitude provider with spec-over-context precedence."""
    if spec.attitude is not None:
        return assemble_attitude_provider(spec.attitude, orbit, universe=context.universe)
    return context.attitude_provider


def _resolve_force_models(spec: PropagatorSpec, context: BuildContext, orbit: Any) -> list[Any]:
    """Resolve numerical force models with spec-over-context precedence."""
    if spec.force_specs:
        sc = spec.spacecraft if spec.spacecraft is not None else SpacecraftSpec()
        return assemble_force_models(spec.force_specs, sc, orbit, universe=context.universe)
    return list(context.force_models)


def _resolve_dsst_force_models(
    spec: PropagatorSpec, context: BuildContext, orbit: Any
) -> list[Any]:
    """Resolve DSST force models with spec-over-context precedence."""
    if spec.force_specs:
        sc = spec.spacecraft if spec.spacecraft is not None else SpacecraftSpec()
        return assemble_dsst_force_models(spec.force_specs, sc, orbit, universe=context.universe)
    return list(context.force_models)


def _position_angle_type(name: str):
    """Resolve an Orekit ``PositionAngleType`` enum by name."""
    try:
        from org.orekit.orbits import PositionAngleType
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc
    return getattr(PositionAngleType, name)


def _propagation_type(name: str):
    """Resolve an Orekit ``PropagationType`` enum by name."""
    try:
        from org.orekit.propagation import PropagationType
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc
    return getattr(PropagationType, name)


@dataclass(frozen=True, slots=True)
class NumericalOrekitProvider:
    """Orekit-native numerical propagator provider."""

    kind: PropagatorKind = PropagatorKind.NUMERICAL
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=True,
        supports_propagator=True,
        supports_stm=True,
    )

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit ``NumericalPropagatorBuilder``.

        Args:
            spec: Propagator spec (requires ``integrator``).
            context: Runtime build context.

        Returns:
            Configured Orekit numerical propagator builder.
        """
        from org.orekit.propagation.conversion import NumericalPropagatorBuilder

        orbit = context.require_initial_orbit()
        integrator_spec = spec.integrator
        if integrator_spec is None:
            raise ValueError("integrator is required for numerical propagation.")
        integrator_builder = create_orekit_integrator_builder(integrator_spec)
        position_angle_type = _position_angle_type(spec.position_angle_type)

        builder = NumericalPropagatorBuilder(
            orbit,
            integrator_builder,
            position_angle_type,
            context.position_tolerance,
        )

        mass = spec.spacecraft.mass if spec.spacecraft else spec.mass_kg
        builder.setMass(mass)

        # Attitude: typed spec > raw context > nothing
        attitude = _resolve_attitude(spec, context, orbit)
        if attitude is not None:
            builder.setAttitudeProvider(attitude)

        # Force models: typed specs > raw context
        for fm in _resolve_force_models(spec, context, orbit):
            builder.addForceModel(fm)

        return builder

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit numerical propagator directly from the builder lane."""
        builder = self.build_builder(spec, context)
        return builder.buildPropagator(builder.getSelectedNormalizedParameters())


@dataclass(frozen=True, slots=True)
class KeplerianOrekitProvider:
    """Orekit-native Keplerian propagator provider."""

    kind: PropagatorKind = PropagatorKind.KEPLERIAN
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=True,
        supports_propagator=True,
    )

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit ``KeplerianPropagatorBuilder``."""
        from org.orekit.propagation.conversion import KeplerianPropagatorBuilder

        orbit = context.require_initial_orbit()
        position_angle_type = _position_angle_type(spec.position_angle_type)

        mu = get_mu(context.universe) if context.universe is not None else orbit.getMu()
        builder = KeplerianPropagatorBuilder(orbit, position_angle_type, mu)

        mass = spec.spacecraft.mass if spec.spacecraft else spec.mass_kg
        builder.setMass(mass)

        attitude = _resolve_attitude(spec, context, orbit)
        if attitude is not None:
            builder.setAttitudeProvider(attitude)

        return builder

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit Keplerian propagator directly from the builder lane."""
        builder = self.build_builder(spec, context)
        return builder.buildPropagator(builder.getSelectedNormalizedParameters())


@dataclass(frozen=True, slots=True)
class DSSTOrekitProvider:
    """Orekit-native DSST propagator provider."""

    kind: PropagatorKind = PropagatorKind.DSST
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=True,
        supports_propagator=True,
        supports_stm=True,
    )

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit ``DSSTPropagatorBuilder`` with DSST force assembly."""
        from org.orekit.propagation.conversion import DSSTPropagatorBuilder

        orbit = context.require_initial_orbit()
        integrator_spec = spec.integrator
        if integrator_spec is None:
            raise ValueError("integrator is required for DSST propagation.")
        integrator_builder = create_orekit_integrator_builder(integrator_spec)
        propagation_type = _propagation_type(spec.dsst_propagation_type)
        state_type = _propagation_type(spec.dsst_state_type)

        dsst_builder_cls: Any = DSSTPropagatorBuilder
        builder = dsst_builder_cls(
            orbit,
            integrator_builder,
            context.position_tolerance,
            propagation_type,
            state_type,
        )

        mass = spec.spacecraft.mass if spec.spacecraft else spec.mass_kg
        builder.setMass(mass)

        # Attitude: typed spec > raw context > nothing
        attitude = _resolve_attitude(spec, context, orbit)
        if attitude is not None:
            builder.setAttitudeProvider(attitude)

        # DSST force models: same ForceSpec types, translated to DSSTForceModel instances
        for fm in _resolve_dsst_force_models(spec, context, orbit):
            builder.addForceModel(fm)

        return builder

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit DSST propagator directly from the builder lane."""
        builder = self.build_builder(spec, context)
        return builder.buildPropagator(builder.getSelectedNormalizedParameters())


@dataclass(frozen=True, slots=True)
class TLEOrekitProvider:
    """Orekit-native TLE/SGP4 propagator provider."""

    kind: PropagatorKind = PropagatorKind.TLE
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=True,
        supports_propagator=True,
        supports_bounded_output=False,
    )

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit ``TLEPropagatorBuilder`` from raw TLE lines."""
        from org.orekit.propagation.analytical.tle import TLE
        from org.orekit.propagation.conversion import TLEPropagatorBuilder

        if spec.tle is None:
            raise ValueError("TLE data is required to build a TLE propagator builder.")

        tle = TLE(spec.tle.line1, spec.tle.line2)
        tle_builder_cls: Any = TLEPropagatorBuilder
        return tle_builder_cls(
            tle,
            _position_angle_type(spec.position_angle_type),
            context.position_tolerance,
            None,  # DataContext — use default
        )

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build an Orekit TLE propagator directly from raw TLE lines."""
        from org.orekit.propagation.analytical.tle import TLE, TLEPropagator

        if spec.tle is None:
            raise ValueError("TLE data is required to build a TLE propagator.")

        tle = TLE(spec.tle.line1, spec.tle.line2)
        return TLEPropagator.selectExtrapolator(tle)


def register_default_orekit_providers(registry: ProviderRegistry) -> None:
    """Register built-in Orekit-native providers in both registry lanes.

    Args:
        registry: Provider registry to mutate.
    """

    providers = [
        NumericalOrekitProvider(),
        KeplerianOrekitProvider(),
        DSSTOrekitProvider(),
        TLEOrekitProvider(),
    ]
    for provider in providers:
        registry.register_builder_provider(provider)
        registry.register_propagator_provider(provider)
