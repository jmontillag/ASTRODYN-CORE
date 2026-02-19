"""Serializable state-file models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from astrodyn_core.states.validation import normalize_orbit_state


@dataclass(frozen=True, slots=True)
class OrbitStateRecord:
    """Single orbital state at one epoch."""

    epoch: str
    frame: str = "GCRF"
    representation: str = "keplerian"
    position_m: Sequence[float] | None = None
    velocity_mps: Sequence[float] | None = None
    elements: Mapping[str, Any] | None = None
    mu_m3_s2: float | str = "WGS84"
    mass_kg: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = normalize_orbit_state(
            epoch=self.epoch,
            frame=self.frame,
            representation=self.representation,
            position_m=self.position_m,
            velocity_mps=self.velocity_mps,
            elements=self.elements,
            mu_m3_s2=self.mu_m3_s2,
            mass_kg=self.mass_kg,
        )
        object.__setattr__(self, "frame", normalized["frame"])
        object.__setattr__(self, "representation", normalized["representation"])
        object.__setattr__(self, "position_m", normalized["position_m"])
        object.__setattr__(self, "velocity_mps", normalized["velocity_mps"])
        object.__setattr__(self, "elements", normalized["elements"])
        object.__setattr__(self, "mu_m3_s2", normalized["mu_m3_s2"])
        object.__setattr__(self, "mass_kg", normalized["mass_kg"])
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> OrbitStateRecord:
        """Build an OrbitStateRecord from a parsed mapping."""
        return cls(
            epoch=str(data["epoch"]),
            frame=data.get("frame", "GCRF"),
            representation=data.get("representation", "keplerian"),
            position_m=data.get("position_m"),
            velocity_mps=data.get("velocity_mps"),
            elements=data.get("elements"),
            mu_m3_s2=data.get("mu_m3_s2", "WGS84"),
            mass_kg=data.get("mass_kg"),
            metadata=data.get("metadata", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Convert this record to a plain serializable dictionary."""
        payload: dict[str, Any] = {
            "epoch": self.epoch,
            "frame": self.frame,
            "representation": self.representation,
            "mu_m3_s2": self.mu_m3_s2,
        }
        if self.representation == "cartesian":
            payload["position_m"] = list(self.position_m or ())
            payload["velocity_mps"] = list(self.velocity_mps or ())
        else:
            payload["elements"] = dict(self.elements or {})
        if self.mass_kg is not None:
            payload["mass_kg"] = self.mass_kg
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class StateSeries:
    """Ordered state timeline for future bounded/interpolated use."""

    name: str
    states: Sequence[OrbitStateRecord]
    interpolation_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("StateSeries.name cannot be empty.")
        if not self.states:
            raise ValueError("StateSeries.states cannot be empty.")
        object.__setattr__(self, "states", tuple(self.states))
        for record in self.states:
            if not isinstance(record, OrbitStateRecord):
                raise TypeError("StateSeries.states must contain OrbitStateRecord values.")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StateSeries:
        states = tuple(OrbitStateRecord.from_mapping(item) for item in data.get("states", []))
        return cls(
            name=str(data.get("name", "series")),
            states=states,
            interpolation_hint=data.get("interpolation_hint"),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {"name": self.name, "states": [record.to_mapping() for record in self.states]}
        if self.interpolation_hint:
            payload["interpolation_hint"] = self.interpolation_hint
        return payload


@dataclass(frozen=True, slots=True)
class ManeuverRecord:
    """Serializable maneuver placeholder for future execution support."""

    name: str
    trigger: Mapping[str, Any]
    model: Mapping[str, Any]
    frame: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ManeuverRecord.name cannot be empty.")
        if not isinstance(self.trigger, Mapping):
            raise TypeError("ManeuverRecord.trigger must be a mapping.")
        if not isinstance(self.model, Mapping):
            raise TypeError("ManeuverRecord.model must be a mapping.")
        object.__setattr__(self, "trigger", dict(self.trigger))
        object.__setattr__(self, "model", dict(self.model))
        if self.frame is not None:
            object.__setattr__(self, "frame", str(self.frame).strip().upper())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ManeuverRecord:
        return cls(
            name=str(data.get("name", "maneuver")),
            trigger=data.get("trigger", {}),
            model=data.get("model", {}),
            frame=data.get("frame"),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "trigger": dict(self.trigger),
            "model": dict(self.model),
        }
        if self.frame is not None:
            payload["frame"] = self.frame
        return payload


@dataclass(frozen=True, slots=True)
class AttitudeRecord:
    """Serializable attitude directive placeholder for future timeline support."""

    mode: str
    frame: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mode.strip():
            raise ValueError("AttitudeRecord.mode cannot be empty.")
        object.__setattr__(self, "mode", self.mode.strip().lower())
        if self.frame is not None:
            object.__setattr__(self, "frame", str(self.frame).strip().upper())
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AttitudeRecord:
        return cls(
            mode=str(data.get("mode", "nadir")),
            frame=data.get("frame"),
            params=data.get("params", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {"mode": self.mode}
        if self.frame is not None:
            payload["frame"] = self.frame
        if self.params:
            payload["params"] = dict(self.params)
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioStateFile:
    """Top-level state-file model."""

    schema_version: int = 1
    universe: Mapping[str, Any] | None = None
    spacecraft: Mapping[str, Any] | None = None
    initial_state: OrbitStateRecord | None = None
    state_series: Sequence[StateSeries] = field(default_factory=tuple)
    maneuvers: Sequence[ManeuverRecord] = field(default_factory=tuple)
    attitude_timeline: Sequence[AttitudeRecord] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported schema_version. Only version 1 is currently supported.")
        if self.initial_state is None and not self.state_series:
            raise ValueError("ScenarioStateFile requires initial_state or at least one state_series.")

        if self.universe is not None and not isinstance(self.universe, Mapping):
            raise TypeError("ScenarioStateFile.universe must be a mapping when provided.")
        if self.spacecraft is not None and not isinstance(self.spacecraft, Mapping):
            raise TypeError("ScenarioStateFile.spacecraft must be a mapping when provided.")
        if self.initial_state is not None and not isinstance(self.initial_state, OrbitStateRecord):
            raise TypeError("ScenarioStateFile.initial_state must be an OrbitStateRecord.")

        object.__setattr__(self, "state_series", tuple(self.state_series))
        object.__setattr__(self, "maneuvers", tuple(self.maneuvers))
        object.__setattr__(self, "attitude_timeline", tuple(self.attitude_timeline))
        object.__setattr__(self, "metadata", dict(self.metadata))

        for item in self.state_series:
            if not isinstance(item, StateSeries):
                raise TypeError("state_series must contain StateSeries values.")
        for item in self.maneuvers:
            if not isinstance(item, ManeuverRecord):
                raise TypeError("maneuvers must contain ManeuverRecord values.")
        for item in self.attitude_timeline:
            if not isinstance(item, AttitudeRecord):
                raise TypeError("attitude_timeline must contain AttitudeRecord values.")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ScenarioStateFile:
        initial_state_data = data.get("initial_state")
        initial_state = (
            OrbitStateRecord.from_mapping(initial_state_data)
            if isinstance(initial_state_data, Mapping)
            else None
        )
        state_series = tuple(StateSeries.from_mapping(item) for item in data.get("state_series", []))
        maneuvers = tuple(ManeuverRecord.from_mapping(item) for item in data.get("maneuvers", []))
        attitude_timeline = tuple(
            AttitudeRecord.from_mapping(item) for item in data.get("attitude_timeline", [])
        )

        return cls(
            schema_version=int(data.get("schema_version", 1)),
            universe=data.get("universe"),
            spacecraft=data.get("spacecraft"),
            initial_state=initial_state,
            state_series=state_series,
            maneuvers=maneuvers,
            attitude_timeline=attitude_timeline,
            metadata=data.get("metadata", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": self.schema_version}
        if self.universe is not None:
            payload["universe"] = dict(self.universe)
        if self.spacecraft is not None:
            payload["spacecraft"] = dict(self.spacecraft)
        if self.initial_state is not None:
            payload["initial_state"] = self.initial_state.to_mapping()
        if self.state_series:
            payload["state_series"] = [series.to_mapping() for series in self.state_series]
        if self.maneuvers:
            payload["maneuvers"] = [item.to_mapping() for item in self.maneuvers]
        if self.attitude_timeline:
            payload["attitude_timeline"] = [item.to_mapping() for item in self.attitude_timeline]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
