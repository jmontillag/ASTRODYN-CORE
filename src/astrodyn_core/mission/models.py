"""Mission domain models used across planning and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CompiledManeuver:
    """Resolved maneuver execution entry.

    Attributes:
        name: Maneuver name from the scenario file.
        trigger_type: Trigger type that resolved the epoch.
        epoch: Resolved execution epoch in ISO-8601 UTC format.
        dv_inertial_mps: Inertial delta-v vector in m/s.
        metadata: Auxiliary metadata from trigger/model resolution.
    """

    name: str
    trigger_type: str
    epoch: str
    dv_inertial_mps: tuple[float, float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedTimelineEvent:
    """Resolved timeline event epoch and source type.

    Attributes:
        id: Timeline event identifier.
        event_type: Resolved event type (epoch/elapsed/apogee/etc.).
        epoch: Resolved event epoch in ISO-8601 UTC format.
    """

    id: str
    event_type: str
    epoch: str
