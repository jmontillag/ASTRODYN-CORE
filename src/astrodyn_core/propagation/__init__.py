"""Propagation core public API."""

from astrodyn_core.propagation.assembly import assemble_attitude_provider, assemble_force_models
from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.capabilities import CapabilityDescriptor
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
    "AttitudeSpec",
    "BuildContext",
    "CapabilityDescriptor",
    "DragSpec",
    "ForceSpec",
    "GravitySpec",
    "IntegratorSpec",
    "OceanTidesSpec",
    "PropagatorFactory",
    "PropagatorKind",
    "PropagatorSpec",
    "ProviderRegistry",
    "RelativitySpec",
    "SRPSpec",
    "SolidTidesSpec",
    "SpacecraftSpec",
    "TLESpec",
    "ThirdBodySpec",
    "assemble_attitude_provider",
    "assemble_force_models",
    "register_default_orekit_providers",
]
