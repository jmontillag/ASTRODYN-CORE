"""Propagation core public API.

Public API
----------
PropagationClient          Facade for ergonomic builder/propagator workflows.
PropagatorFactory          Low-level factory for building propagators from specs.
ProviderRegistry           Registry of propagation backend providers.
BuildContext               Context for propagator construction (orbit, universe, metadata).

Specs and models:
    PropagatorSpec, PropagatorKind, IntegratorSpec, TLESpec,
    AttitudeSpec, SpacecraftSpec,
    ForceSpec, GravitySpec, DragSpec, SRPSpec, ThirdBodySpec,
    RelativitySpec, SolidTidesSpec, OceanTidesSpec,
    CapabilityDescriptor

Assembly and config helpers:
    assemble_force_models, assemble_attitude_provider,
    load_dynamics_config, load_dynamics_from_dict,
    load_spacecraft_config, load_spacecraft_from_dict,
    register_default_orekit_providers

Universe config helpers (available via ``propagation.config`` or ``propagation.universe``):
    get_mu, get_earth_shape, get_iers_conventions, get_itrf_frame, etc.
"""

from astrodyn_core.propagation.assembly import assemble_attitude_provider, assemble_force_models
from astrodyn_core.propagation.dsst_assembly import assemble_dsst_force_models
from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.client import PropagationClient
from astrodyn_core.propagation.config import (
    load_dynamics_config,
    load_dynamics_from_dict,
    load_spacecraft_config,
    load_spacecraft_from_dict,
)
from astrodyn_core.propagation.factory import PropagatorFactory
from astrodyn_core.propagation.forces import (
    DragSpec,
    ForceSpec,
    GravitySpec,
    OceanTidesSpec,
    RelativitySpec,
    SRPSpec,
    SolidTidesSpec,
    ThirdBodySpec,
)
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.providers.orekit_native import register_default_orekit_providers
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.spacecraft import SpacecraftSpec
from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec, TLESpec

__all__ = [
    # Facade
    "PropagationClient",
    # Factory / Registry
    "PropagatorFactory",
    "ProviderRegistry",
    "register_default_orekit_providers",
    # Specs and models
    "PropagatorSpec",
    "PropagatorKind",
    "IntegratorSpec",
    "TLESpec",
    "AttitudeSpec",
    "SpacecraftSpec",
    "CapabilityDescriptor",
    "BuildContext",
    # Force specs
    "ForceSpec",
    "GravitySpec",
    "DragSpec",
    "SRPSpec",
    "ThirdBodySpec",
    "RelativitySpec",
    "SolidTidesSpec",
    "OceanTidesSpec",
    # Assembly and config
    "assemble_force_models",
    "assemble_dsst_force_models",
    "assemble_attitude_provider",
    "load_dynamics_config",
    "load_dynamics_from_dict",
    "load_spacecraft_config",
    "load_spacecraft_from_dict",
]
