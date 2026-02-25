"""Serializable state-file models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence

from astrodyn_core.states.validation import normalize_orbit_state, parse_epoch_utc


@dataclass(frozen=True, slots=True)
class OrbitStateRecord:
    """Serializable orbital state snapshot at one epoch.

    Attributes:
        epoch: ISO-8601 UTC epoch string.
        frame: Reference frame name (normalized to uppercase).
        representation: State representation (``cartesian``, ``keplerian``, or
            ``equinoctial``; normalized to lowercase).
        position_m: Cartesian position vector in meters (cartesian only).
        velocity_mps: Cartesian velocity vector in m/s (cartesian only).
        elements: Orbital elements mapping (keplerian/equinoctial only).
        mu_m3_s2: Gravitational parameter as numeric value or symbolic model key.
        mass_kg: Optional spacecraft mass in kg.
        metadata: Free-form metadata preserved through serialization.
    """

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
        """Build an orbit-state record from a parsed mapping.

        Args:
            data: Mapping loaded from YAML/JSON.

        Returns:
            Validated and normalized orbit-state record.
        """
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
        """Serialize this record to a plain mapping.

        Returns:
            Mapping suitable for YAML/JSON serialization.
        """
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
class OutputEpochSpec:
    """Epoch grid specification for trajectory export and sampling helpers.

    The grid can be defined in one of two mutually exclusive modes:

    - explicit epoch list via ``explicit_epochs``
    - generated range via ``start_epoch``/``end_epoch`` and either
      ``step_seconds`` or ``count``

    Attributes:
        explicit_epochs: Explicit ISO-8601 UTC epochs.
        start_epoch: Inclusive range start epoch.
        end_epoch: Inclusive range end epoch.
        step_seconds: Fixed step size in seconds (range mode only).
        count: Number of output epochs (range mode only).
        include_end: Whether generated range output should include ``end_epoch``.
    """

    explicit_epochs: Sequence[str] = field(default_factory=tuple)
    start_epoch: str | None = None
    end_epoch: str | None = None
    step_seconds: float | None = None
    count: int | None = None
    include_end: bool = True

    def __post_init__(self) -> None:
        explicit = tuple(self.explicit_epochs)
        object.__setattr__(self, "explicit_epochs", explicit)

        if explicit:
            if any(
                value is not None
                for value in (self.start_epoch, self.end_epoch, self.step_seconds, self.count)
            ):
                raise ValueError(
                    "OutputEpochSpec cannot mix explicit_epochs with start/end/step/count fields."
                )
            if len(explicit) == 0:
                raise ValueError("OutputEpochSpec.explicit_epochs cannot be empty.")
            for epoch in explicit:
                parse_epoch_utc(str(epoch))
            return

        if self.start_epoch is None or self.end_epoch is None:
            raise ValueError("OutputEpochSpec requires start_epoch and end_epoch when explicit_epochs is not set.")
        parse_epoch_utc(self.start_epoch)
        parse_epoch_utc(self.end_epoch)
        if parse_epoch_utc(self.end_epoch) < parse_epoch_utc(self.start_epoch):
            raise ValueError("OutputEpochSpec.end_epoch must be >= start_epoch.")

        if (self.step_seconds is None) == (self.count is None):
            raise ValueError("OutputEpochSpec requires exactly one of step_seconds or count.")

        if self.step_seconds is not None and float(self.step_seconds) <= 0:
            raise ValueError("OutputEpochSpec.step_seconds must be positive.")

        if self.count is not None and int(self.count) < 2:
            raise ValueError("OutputEpochSpec.count must be >= 2.")

    def epochs(self) -> tuple[str, ...]:
        """Return the concrete output epochs as ISO-8601 UTC strings.

        Returns:
            Tuple of UTC epoch strings in output order.
        """
        if self.explicit_epochs:
            return tuple(str(epoch) for epoch in self.explicit_epochs)

        start = parse_epoch_utc(self.start_epoch or "")
        end = parse_epoch_utc(self.end_epoch or "")

        if self.step_seconds is not None:
            step = timedelta(seconds=float(self.step_seconds))
            epochs: list[str] = []
            current = start
            while current < end:
                epochs.append(current.isoformat().replace("+00:00", "Z"))
                current = current + step
            if self.include_end and (not epochs or parse_epoch_utc(epochs[-1]) != end):
                epochs.append(end.isoformat().replace("+00:00", "Z"))
            return tuple(epochs)

        count = int(self.count or 0)
        span_s = (end - start).total_seconds()
        step_s = span_s / float(count - 1)
        epochs = []
        for idx in range(count):
            current = start + timedelta(seconds=idx * step_s)
            epochs.append(current.isoformat().replace("+00:00", "Z"))
        if self.include_end and epochs:
            epochs[-1] = end.isoformat().replace("+00:00", "Z")
        return tuple(epochs)


@dataclass(frozen=True, slots=True)
class StateSeries:
    """Ordered sequence of orbit states with optional interpolation metadata.

    Attributes:
        name: Series identifier.
        states: Ordered orbit-state records.
        interpolation_hint: Informal interpolation hint label.
        interpolation: Structured interpolation metadata mapping.
    """

    name: str
    states: Sequence[OrbitStateRecord]
    interpolation_hint: str | None = None
    interpolation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("StateSeries.name cannot be empty.")
        if not self.states:
            raise ValueError("StateSeries.states cannot be empty.")
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "interpolation", dict(self.interpolation))
        for record in self.states:
            if not isinstance(record, OrbitStateRecord):
                raise TypeError("StateSeries.states must contain OrbitStateRecord values.")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StateSeries:
        """Build a state series from expanded or compact serialized mappings.

        Args:
            data: Mapping containing either ``states`` or compact ``rows`` data.

        Returns:
            Validated state series.
        """
        interpolation = data.get("interpolation", {})
        if not isinstance(interpolation, Mapping):
            raise TypeError("StateSeries.interpolation must be a mapping when provided.")
        interpolation_hint = data.get("interpolation_hint")
        if interpolation_hint is None and "method" in interpolation:
            interpolation_hint = str(interpolation["method"])

        if "states" in data:
            states = tuple(OrbitStateRecord.from_mapping(item) for item in data.get("states", []))
        elif "rows" in data:
            states = _parse_compact_state_rows(data)
        else:
            states = ()
        return cls(
            name=str(data.get("name", "series")),
            states=states,
            interpolation_hint=interpolation_hint,
            interpolation=interpolation,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the state series into an expanded mapping.

        Returns:
            Mapping with ``states`` entries suitable for YAML/JSON serialization.
        """
        payload = {"name": self.name, "states": [record.to_mapping() for record in self.states]}
        if self.interpolation_hint:
            payload["interpolation_hint"] = self.interpolation_hint
        if self.interpolation:
            payload["interpolation"] = dict(self.interpolation)
        return payload


@dataclass(frozen=True, slots=True)
class ManeuverRecord:
    """Serializable maneuver placeholder for scenario timelines.

    Attributes:
        name: Maneuver identifier.
        trigger: Trigger condition mapping.
        model: Maneuver model/parameter mapping.
        frame: Optional frame name (normalized to uppercase).
    """

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
        """Build a maneuver record from a plain mapping.

        Args:
            data: Serialized maneuver mapping.

        Returns:
            Validated maneuver record.
        """
        return cls(
            name=str(data.get("name", "maneuver")),
            trigger=data.get("trigger", {}),
            model=data.get("model", {}),
            frame=data.get("frame"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the maneuver record to a plain mapping.

        Returns:
            Mapping suitable for scenario-file serialization.
        """
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
    """Serializable attitude directive placeholder for timeline support.

    Attributes:
        mode: Attitude mode name (normalized to lowercase).
        frame: Optional reference frame (normalized to uppercase).
        params: Free-form attitude parameters.
    """

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
        """Build an attitude record from a plain mapping.

        Args:
            data: Serialized attitude mapping.

        Returns:
            Validated attitude record.
        """
        return cls(
            mode=str(data.get("mode", "nadir")),
            frame=data.get("frame"),
            params=data.get("params", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the attitude record to a plain mapping.

        Returns:
            Mapping suitable for scenario-file serialization.
        """
        payload = {"mode": self.mode}
        if self.frame is not None:
            payload["frame"] = self.frame
        if self.params:
            payload["params"] = dict(self.params)
        return payload


@dataclass(frozen=True, slots=True)
class TimelineEventRecord:
    """Named timeline event used to schedule mission actions.

    Attributes:
        id: Unique event identifier.
        point: Event point definition mapping.
        metadata: Free-form event metadata mapping.
    """

    id: str
    point: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = self.id.strip()
        if not event_id:
            raise ValueError("TimelineEventRecord.id cannot be empty.")
        if not isinstance(self.point, Mapping):
            raise TypeError("TimelineEventRecord.point must be a mapping.")
        object.__setattr__(self, "id", event_id)
        object.__setattr__(self, "point", dict(self.point))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TimelineEventRecord:
        """Build a timeline event record from a plain mapping.

        Args:
            data: Serialized event mapping.

        Returns:
            Validated timeline event record.
        """
        return cls(
            id=str(data.get("id", "")).strip(),
            point=data.get("point", {}),
            metadata=data.get("metadata", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the timeline event record to a plain mapping.

        Returns:
            Mapping suitable for scenario-file serialization.
        """
        payload = {
            "id": self.id,
            "point": dict(self.point),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioStateFile:
    """Top-level scenario state-file model.

    Attributes:
        schema_version: Scenario schema version (currently ``1`` only).
        universe: Optional universe configuration mapping.
        spacecraft: Optional spacecraft configuration mapping.
        initial_state: Optional initial state record.
        timeline: Timeline events for mission sequencing.
        state_series: Serialized state trajectories.
        maneuvers: Maneuver definitions.
        attitude_timeline: Attitude directives.
        metadata: Free-form scenario metadata.
    """

    schema_version: int = 1
    universe: Mapping[str, Any] | None = None
    spacecraft: Mapping[str, Any] | None = None
    initial_state: OrbitStateRecord | None = None
    timeline: Sequence[TimelineEventRecord] = field(default_factory=tuple)
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

        object.__setattr__(self, "timeline", tuple(self.timeline))
        object.__setattr__(self, "state_series", tuple(self.state_series))
        object.__setattr__(self, "maneuvers", tuple(self.maneuvers))
        object.__setattr__(self, "attitude_timeline", tuple(self.attitude_timeline))
        object.__setattr__(self, "metadata", dict(self.metadata))

        seen_event_ids: set[str] = set()
        for item in self.timeline:
            if not isinstance(item, TimelineEventRecord):
                raise TypeError("timeline must contain TimelineEventRecord values.")
            if item.id in seen_event_ids:
                raise ValueError(f"timeline contains duplicate event id '{item.id}'.")
            seen_event_ids.add(item.id)
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
        """Build a scenario state file from a serialized mapping.

        Args:
            data: Mapping loaded from YAML/JSON.

        Returns:
            Validated scenario state file model.
        """
        initial_state_data = data.get("initial_state")
        initial_state = (
            OrbitStateRecord.from_mapping(initial_state_data)
            if isinstance(initial_state_data, Mapping)
            else None
        )
        timeline = tuple(TimelineEventRecord.from_mapping(item) for item in data.get("timeline", []))
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
            timeline=timeline,
            state_series=state_series,
            maneuvers=maneuvers,
            attitude_timeline=attitude_timeline,
            metadata=data.get("metadata", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the scenario state file to a plain mapping.

        Returns:
            Mapping suitable for YAML/JSON serialization.
        """
        payload: dict[str, Any] = {"schema_version": self.schema_version}
        if self.universe is not None:
            payload["universe"] = dict(self.universe)
        if self.spacecraft is not None:
            payload["spacecraft"] = dict(self.spacecraft)
        if self.initial_state is not None:
            payload["initial_state"] = self.initial_state.to_mapping()
        if self.timeline:
            payload["timeline"] = [item.to_mapping() for item in self.timeline]
        if self.state_series:
            payload["state_series"] = [series.to_mapping() for series in self.state_series]
        if self.maneuvers:
            payload["maneuvers"] = [item.to_mapping() for item in self.maneuvers]
        if self.attitude_timeline:
            payload["attitude_timeline"] = [item.to_mapping() for item in self.attitude_timeline]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _parse_compact_state_rows(data: Mapping[str, Any]) -> tuple[OrbitStateRecord, ...]:
    rows = data.get("rows", [])
    defaults_raw = data.get("defaults", {})
    columns_raw = data.get("columns")

    if not isinstance(defaults_raw, Mapping):
        raise TypeError("state_series.defaults must be a mapping when provided.")
    defaults = dict(defaults_raw)

    columns: list[str] | None = None
    if columns_raw is not None:
        if isinstance(columns_raw, (str, bytes)):
            raise TypeError("state_series.columns must be a sequence of field names.")
        columns = [str(item) for item in columns_raw]
        if not columns:
            raise ValueError("state_series.columns cannot be empty when provided.")

    items: list[OrbitStateRecord] = []
    for idx, row in enumerate(rows):
        row_data: dict[str, Any]
        if isinstance(row, Mapping):
            row_data = dict(row)
        elif columns is not None and not isinstance(row, (str, bytes)):
            if len(row) != len(columns):
                raise ValueError(
                    f"state_series row #{idx} length {len(row)} does not match columns {len(columns)}."
                )
            row_data = dict(zip(columns, row))
        else:
            raise TypeError(
                "Each state_series row must be a mapping, or a sequence when columns are provided."
            )

        merged = _merge_state_row(defaults, row_data)
        merged = _normalize_compact_state_shape(merged)
        items.append(OrbitStateRecord.from_mapping(merged))

    return tuple(items)


def _merge_state_row(defaults: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    base_elements = merged.get("elements")
    if isinstance(base_elements, Mapping):
        merged["elements"] = dict(base_elements)

    for key, value in row.items():
        key_str = str(key)
        if "." in key_str:
            parts = key_str.split(".")
            _set_dotted(merged, parts, value)
            continue

        if key_str == "elements" and isinstance(value, Mapping):
            current = merged.get("elements")
            if isinstance(current, Mapping):
                next_elements = dict(current)
                next_elements.update(dict(value))
                merged["elements"] = next_elements
            else:
                merged["elements"] = dict(value)
            continue

        merged[key_str] = value

    return merged


def _set_dotted(target: dict[str, Any], path: Sequence[str], value: Any) -> None:
    node: dict[str, Any] = target
    for part in path[:-1]:
        existing = node.get(part)
        if not isinstance(existing, dict):
            existing = {}
            node[part] = existing
        node = existing
    node[path[-1]] = value


def _normalize_compact_state_shape(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    representation = str(data.get("representation", "")).strip().lower()

    if representation == "cartesian":
        if "position_m" not in data and all(key in data for key in ("x_m", "y_m", "z_m")):
            data["position_m"] = [data.pop("x_m"), data.pop("y_m"), data.pop("z_m")]
        if "velocity_mps" not in data and all(key in data for key in ("vx_mps", "vy_mps", "vz_mps")):
            data["velocity_mps"] = [data.pop("vx_mps"), data.pop("vy_mps"), data.pop("vz_mps")]
        return data

    if representation == "keplerian":
        if "elements" not in data:
            keys = ("a_m", "e", "i_deg", "argp_deg", "raan_deg", "anomaly_deg", "anomaly_type")
            if any(key in data for key in keys):
                data["elements"] = {}
            for key in keys:
                if key in data:
                    data["elements"][key] = data.pop(key)
        return data

    if representation == "equinoctial":
        if "elements" not in data:
            keys = ("a_m", "ex", "ey", "hx", "hy", "l_deg", "anomaly_type")
            if any(key in data for key in keys):
                data["elements"] = {}
            for key in keys:
                if key in data:
                    data["elements"][key] = data.pop(key)
        return data

    return data
