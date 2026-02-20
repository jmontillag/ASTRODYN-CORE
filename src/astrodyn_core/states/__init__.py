"""State-file models and conversion helpers.

Public API
----------
StateFileClient        Facade for state-file workflows and Orekit conversion.
OrbitStateRecord       Single orbital state at one epoch.
StateSeries            Ordered sequence of orbital states.
ScenarioStateFile      Top-level scenario container (state + maneuvers + timeline).
OutputEpochSpec        Epoch grid specification for trajectory export.
ManeuverRecord         Maneuver definition from scenario file.
TimelineEventRecord    Named timeline event definition.
AttitudeRecord         Attitude mode record from timeline.
parse_epoch_utc        Parse ISO-8601 epoch string to UTC datetime.
"""

from astrodyn_core.states.client import StateFileClient
from astrodyn_core.states.models import (
    AttitudeRecord,
    ManeuverRecord,
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
    TimelineEventRecord,
)
from astrodyn_core.states.validation import parse_epoch_utc

__all__ = [
    "StateFileClient",
    "OrbitStateRecord",
    "StateSeries",
    "ScenarioStateFile",
    "OutputEpochSpec",
    "ManeuverRecord",
    "TimelineEventRecord",
    "AttitudeRecord",
    "parse_epoch_utc",
]
