"""ASTRODYN-CORE public package API.

Exports are organized into three tiers:

Tier 1 -- Facade clients (recommended for most users)
    AstrodynClient, PropagationClient, StateFileClient, MissionClient,
    UncertaintyClient, TLEClient

Tier 2 -- Data models and specs (needed for configuration)
    PropagatorSpec, PropagatorKind, IntegratorSpec, BuildContext,
    force/spacecraft/attitude specs, state/scenario models,
    mission models, uncertainty models, TLE models

Tier 3 -- Advanced low-level helpers (expert Orekit-native usage)
    PropagatorFactory, ProviderRegistry, register_default_orekit_providers,
    assembly helpers, config loaders, data preset lookups

All Tier 1-3 symbols require orekit to be installed (``conda install orekit``).
Subpackages like ``astrodyn_core.geqoe_taylor`` work without orekit.

Symbols that were previously exported at root but belong to domain-specific
workflows (e.g. compile_scenario_maneuvers, setup_stm_propagator, individual
TLE parser functions) have been moved to subpackage exports only.  They remain
importable via their subpackage paths (e.g. ``from astrodyn_core.mission import
compile_scenario_maneuvers``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrodyn_core.client import AstrodynClient
    from astrodyn_core.data import (
        get_propagation_model,
        get_spacecraft_model,
        list_propagation_models,
        list_spacecraft_models,
    )
    from astrodyn_core.ephemeris import EphemerisClient, EphemerisFormat, EphemerisSpec
    from astrodyn_core.mission import (
        CompiledManeuver,
        ManeuverFiredEvent,
        MissionClient,
        MissionExecutionReport,
        ScenarioExecutor,
    )
    from astrodyn_core.propagation import (
        AttitudeSpec,
        BuildContext,
        CapabilityDescriptor,
        DragSpec,
        ForceSpec,
        GravitySpec,
        IntegratorSpec,
        OceanTidesSpec,
        PropagationClient,
        PropagatorFactory,
        PropagatorKind,
        PropagatorSpec,
        ProviderRegistry,
        RelativitySpec,
        SRPSpec,
        SolidTidesSpec,
        SpacecraftSpec,
        TLESpec,
        ThirdBodySpec,
        assemble_attitude_provider,
        assemble_force_models,
        load_dynamics_config,
        load_dynamics_from_dict,
        load_spacecraft_config,
        load_spacecraft_from_dict,
        register_default_orekit_providers,
    )
    from astrodyn_core.states import (
        AttitudeRecord,
        ManeuverRecord,
        OrbitStateRecord,
        OutputEpochSpec,
        ScenarioStateFile,
        StateFileClient,
        StateSeries,
        TimelineEventRecord,
    )
    from astrodyn_core.tle import TLEClient, TLEDownloadResult, TLEQuery, TLERecord
    from astrodyn_core.uncertainty import (
        CovarianceRecord,
        CovarianceSeries,
        UncertaintyClient,
        UncertaintySpec,
    )

# ---------------------------------------------------------------------------
# Lazy import map: attribute name -> (module, name)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # -- Tier 1: Facade clients --
    "AstrodynClient": ("astrodyn_core.client", "AstrodynClient"),
    "PropagationClient": ("astrodyn_core.propagation", "PropagationClient"),
    "StateFileClient": ("astrodyn_core.states", "StateFileClient"),
    "MissionClient": ("astrodyn_core.mission", "MissionClient"),
    "UncertaintyClient": ("astrodyn_core.uncertainty", "UncertaintyClient"),
    "TLEClient": ("astrodyn_core.tle", "TLEClient"),
    "EphemerisClient": ("astrodyn_core.ephemeris", "EphemerisClient"),
    # -- Tier 2: Data models and specs --
    # Propagation
    "PropagatorSpec": ("astrodyn_core.propagation", "PropagatorSpec"),
    "PropagatorKind": ("astrodyn_core.propagation", "PropagatorKind"),
    "IntegratorSpec": ("astrodyn_core.propagation", "IntegratorSpec"),
    "BuildContext": ("astrodyn_core.propagation", "BuildContext"),
    "AttitudeSpec": ("astrodyn_core.propagation", "AttitudeSpec"),
    "CapabilityDescriptor": ("astrodyn_core.propagation", "CapabilityDescriptor"),
    "DragSpec": ("astrodyn_core.propagation", "DragSpec"),
    "ForceSpec": ("astrodyn_core.propagation", "ForceSpec"),
    "GravitySpec": ("astrodyn_core.propagation", "GravitySpec"),
    "OceanTidesSpec": ("astrodyn_core.propagation", "OceanTidesSpec"),
    "RelativitySpec": ("astrodyn_core.propagation", "RelativitySpec"),
    "SRPSpec": ("astrodyn_core.propagation", "SRPSpec"),
    "SolidTidesSpec": ("astrodyn_core.propagation", "SolidTidesSpec"),
    "SpacecraftSpec": ("astrodyn_core.propagation", "SpacecraftSpec"),
    "TLESpec": ("astrodyn_core.propagation", "TLESpec"),
    "ThirdBodySpec": ("astrodyn_core.propagation", "ThirdBodySpec"),
    # States / Scenario
    "OrbitStateRecord": ("astrodyn_core.states", "OrbitStateRecord"),
    "StateSeries": ("astrodyn_core.states", "StateSeries"),
    "ScenarioStateFile": ("astrodyn_core.states", "ScenarioStateFile"),
    "OutputEpochSpec": ("astrodyn_core.states", "OutputEpochSpec"),
    "ManeuverRecord": ("astrodyn_core.states", "ManeuverRecord"),
    "TimelineEventRecord": ("astrodyn_core.states", "TimelineEventRecord"),
    "AttitudeRecord": ("astrodyn_core.states", "AttitudeRecord"),
    # Mission
    "CompiledManeuver": ("astrodyn_core.mission", "CompiledManeuver"),
    "MissionExecutionReport": ("astrodyn_core.mission", "MissionExecutionReport"),
    "ManeuverFiredEvent": ("astrodyn_core.mission", "ManeuverFiredEvent"),
    "ScenarioExecutor": ("astrodyn_core.mission", "ScenarioExecutor"),
    # Uncertainty
    "UncertaintySpec": ("astrodyn_core.uncertainty", "UncertaintySpec"),
    "CovarianceRecord": ("astrodyn_core.uncertainty", "CovarianceRecord"),
    "CovarianceSeries": ("astrodyn_core.uncertainty", "CovarianceSeries"),
    # TLE
    "TLEQuery": ("astrodyn_core.tle", "TLEQuery"),
    "TLERecord": ("astrodyn_core.tle", "TLERecord"),
    "TLEDownloadResult": ("astrodyn_core.tle", "TLEDownloadResult"),
    # Ephemeris
    "EphemerisSpec": ("astrodyn_core.ephemeris", "EphemerisSpec"),
    "EphemerisFormat": ("astrodyn_core.ephemeris", "EphemerisFormat"),
    # -- Tier 3: Advanced low-level --
    "PropagatorFactory": ("astrodyn_core.propagation", "PropagatorFactory"),
    "ProviderRegistry": ("astrodyn_core.propagation", "ProviderRegistry"),
    "register_default_orekit_providers": (
        "astrodyn_core.propagation",
        "register_default_orekit_providers",
    ),
    "assemble_force_models": ("astrodyn_core.propagation", "assemble_force_models"),
    "assemble_attitude_provider": ("astrodyn_core.propagation", "assemble_attitude_provider"),
    "load_dynamics_config": ("astrodyn_core.propagation", "load_dynamics_config"),
    "load_dynamics_from_dict": ("astrodyn_core.propagation", "load_dynamics_from_dict"),
    "load_spacecraft_config": ("astrodyn_core.propagation", "load_spacecraft_config"),
    "load_spacecraft_from_dict": ("astrodyn_core.propagation", "load_spacecraft_from_dict"),
    "get_propagation_model": ("astrodyn_core.data", "get_propagation_model"),
    "get_spacecraft_model": ("astrodyn_core.data", "get_spacecraft_model"),
    "list_propagation_models": ("astrodyn_core.data", "list_propagation_models"),
    "list_spacecraft_models": ("astrodyn_core.data", "list_spacecraft_models"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache on the module so __getattr__ is only called once per name
        globals()[name] = value
        return value
    raise AttributeError(f"module 'astrodyn_core' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# __all__ — curated public API
# ---------------------------------------------------------------------------
__all__ = list(_LAZY_IMPORTS.keys())
