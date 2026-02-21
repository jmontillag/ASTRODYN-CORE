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

Symbols that were previously exported at root but belong to domain-specific
workflows (e.g. compile_scenario_maneuvers, setup_stm_propagator, individual
TLE parser functions) have been moved to subpackage exports only.  They remain
importable via their subpackage paths (e.g. ``from astrodyn_core.mission import
compile_scenario_maneuvers``).
"""

# ---------------------------------------------------------------------------
# Tier 1: Facade clients
# ---------------------------------------------------------------------------
from astrodyn_core.client import AstrodynClient
from astrodyn_core.ephemeris import EphemerisClient
from astrodyn_core.mission import MissionClient
from astrodyn_core.propagation import PropagationClient
from astrodyn_core.states import StateFileClient
from astrodyn_core.tle import TLEClient
from astrodyn_core.uncertainty import UncertaintyClient

# ---------------------------------------------------------------------------
# Tier 2: Data models and specs
# ---------------------------------------------------------------------------

# Propagation specs
from astrodyn_core.propagation import (
    AttitudeSpec,
    BuildContext,
    CapabilityDescriptor,
    DragSpec,
    ForceSpec,
    GravitySpec,
    IntegratorSpec,
    OceanTidesSpec,
    PropagatorKind,
    PropagatorSpec,
    RelativitySpec,
    SRPSpec,
    SolidTidesSpec,
    SpacecraftSpec,
    TLESpec,
    ThirdBodySpec,
)

# State and scenario models
from astrodyn_core.states import (
    AttitudeRecord,
    ManeuverRecord,
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
    TimelineEventRecord,
)

# Mission models
from astrodyn_core.mission import (
    CompiledManeuver,
    ManeuverFiredEvent,
    MissionExecutionReport,
    ScenarioExecutor,
)

# Uncertainty models
from astrodyn_core.uncertainty import (
    CovarianceRecord,
    CovarianceSeries,
    UncertaintySpec,
)

# TLE models
from astrodyn_core.tle import (
    TLEDownloadResult,
    TLEQuery,
    TLERecord,
)

# Ephemeris models
from astrodyn_core.ephemeris import (
    EphemerisFormat,
    EphemerisSpec,
)

# ---------------------------------------------------------------------------
# Tier 3: Advanced low-level helpers
# ---------------------------------------------------------------------------
from astrodyn_core.propagation import (
    PropagatorFactory,
    ProviderRegistry,
    assemble_attitude_provider,
    assemble_force_models,
    load_dynamics_config,
    load_dynamics_from_dict,
    load_spacecraft_config,
    load_spacecraft_from_dict,
    register_default_orekit_providers,
)
from astrodyn_core.data import (
    get_propagation_model,
    get_spacecraft_model,
    list_propagation_models,
    list_spacecraft_models,
)

# ---------------------------------------------------------------------------
# __all__ — curated public API
# ---------------------------------------------------------------------------
__all__ = [
    # -- Tier 1: Facade clients --
    "AstrodynClient",
    "PropagationClient",
    "StateFileClient",
    "MissionClient",
    "UncertaintyClient",
    "TLEClient",
    "EphemerisClient",
    # -- Tier 2: Data models and specs --
    # Propagation
    "PropagatorSpec",
    "PropagatorKind",
    "IntegratorSpec",
    "BuildContext",
    "AttitudeSpec",
    "CapabilityDescriptor",
    "DragSpec",
    "ForceSpec",
    "GravitySpec",
    "OceanTidesSpec",
    "RelativitySpec",
    "SRPSpec",
    "SolidTidesSpec",
    "SpacecraftSpec",
    "TLESpec",
    "ThirdBodySpec",
    # States / Scenario
    "OrbitStateRecord",
    "StateSeries",
    "ScenarioStateFile",
    "OutputEpochSpec",
    "ManeuverRecord",
    "TimelineEventRecord",
    "AttitudeRecord",
    # Mission
    "CompiledManeuver",
    "MissionExecutionReport",
    "ManeuverFiredEvent",
    "ScenarioExecutor",
    # Uncertainty
    "UncertaintySpec",
    "CovarianceRecord",
    "CovarianceSeries",
    # TLE
    "TLEQuery",
    "TLERecord",
    "TLEDownloadResult",
    # Ephemeris
    "EphemerisSpec",
    "EphemerisFormat",
    # -- Tier 3: Advanced low-level --
    "PropagatorFactory",
    "ProviderRegistry",
    "register_default_orekit_providers",
    "assemble_force_models",
    "assemble_attitude_provider",
    "load_dynamics_config",
    "load_dynamics_from_dict",
    "load_spacecraft_config",
    "load_spacecraft_from_dict",
    "get_propagation_model",
    "get_spacecraft_model",
    "list_propagation_models",
    "list_spacecraft_models",
]
