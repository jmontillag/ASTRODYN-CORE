"""Mission-profile helpers (maneuver execution and analysis plotting).

Public API
----------
MissionClient              Facade for mission planning, execution, and plotting.
CompiledManeuver           Compiled maneuver with resolved epoch and delta-v.
MissionExecutionReport     Summary report from detector-driven execution.
ManeuverFiredEvent         Single maneuver event record from execution.
ScenarioExecutor           Detector-driven mission execution engine.
compile_scenario_maneuvers Compile scenario maneuvers from state file.
plot_orbital_elements_series  Plot orbital elements from a state series.
"""

from astrodyn_core.mission.client import MissionClient
from astrodyn_core.mission.executor import (
    ManeuverFiredEvent,
    MissionExecutionReport,
    ScenarioExecutor,
)
from astrodyn_core.mission.models import CompiledManeuver
from astrodyn_core.mission.simulation import compile_scenario_maneuvers
from astrodyn_core.mission.plotting import plot_orbital_elements_series

__all__ = [
    "MissionClient",
    "CompiledManeuver",
    "MissionExecutionReport",
    "ManeuverFiredEvent",
    "ScenarioExecutor",
    "compile_scenario_maneuvers",
    "plot_orbital_elements_series",
]
