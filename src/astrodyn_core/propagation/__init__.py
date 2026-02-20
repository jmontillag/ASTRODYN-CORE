"""Propagation core public API."""

from astrodyn_core.propagation.assembly import assemble_attitude_provider, assemble_force_models
from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.client import PropagationClient
from astrodyn_core.propagation.config import (
    get_earth_shape,
    get_iers_conventions,
    get_itrf_frame,
    get_itrf_version,
    get_mu,
    get_universe_config,
    load_default_universe_config,
    load_dynamics_config,
    load_dynamics_from_dict,
    load_spacecraft_config,
    load_spacecraft_from_dict,
    load_universe_config,
    load_universe_from_dict,
)
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
    "PropagationClient",
    "get_earth_shape",
    "get_iers_conventions",
    "get_itrf_frame",
    "get_itrf_version",
    "get_mu",
    "get_universe_config",
    "load_default_universe_config",
    "load_dynamics_config",
    "load_dynamics_from_dict",
    "load_spacecraft_config",
    "load_spacecraft_from_dict",
    "load_universe_config",
    "load_universe_from_dict",
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
