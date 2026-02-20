"""Mission domain models used across planning and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CompiledManeuver:
    """Resolved maneuver execution entry."""

    name: str
    trigger_type: str
    epoch: str
    dv_inertial_mps: tuple[float, float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedTimelineEvent:
    """Resolved timeline event epoch and source type."""

    id: str
    event_type: str
    epoch: str
