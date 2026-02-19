"""Scenario maneuver planning and execution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from astrodyn_core.states.io import save_state_series_compact_with_style, save_state_series_hdf5
from astrodyn_core.states.models import (
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
    TimelineEventRecord,
)
from astrodyn_core.states.orekit import from_orekit_date, resolve_frame, to_orekit_date


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


def compile_scenario_maneuvers(
    scenario: ScenarioStateFile,
    initial_state: Any,
) -> tuple[CompiledManeuver, ...]:
    """Compile scenario maneuvers to absolute epochs + inertial delta-v vectors.

    Intent solving is intentionally Keplerian-only for speed.
    """
    if not isinstance(scenario, ScenarioStateFile):
        raise TypeError("scenario must be a ScenarioStateFile.")
    if not hasattr(initial_state, "getDate") or not hasattr(initial_state, "getOrbit"):
        raise TypeError("initial_state must be an Orekit SpacecraftState-like object.")

    timeline = _resolve_timeline_events(scenario.timeline, initial_state)
    compiled: list[CompiledManeuver] = []
    planning_state = initial_state

    for maneuver in scenario.maneuvers:
        trigger_type, event_date, trigger_metadata = _resolve_maneuver_trigger(
            maneuver.trigger,
            planning_state,
            timeline,
        )
        event_state = _keplerian_propagate_state(planning_state, event_date)
        dv_vec = _resolve_delta_v_vector(maneuver.model, event_state, trigger_type)

        metadata = {"model_type": str(maneuver.model.get("type", "unknown"))}
        metadata.update(trigger_metadata)
        compiled_entry = CompiledManeuver(
            name=maneuver.name,
            trigger_type=trigger_type,
            epoch=from_orekit_date(event_date),
            dv_inertial_mps=(float(dv_vec.getX()), float(dv_vec.getY()), float(dv_vec.getZ())),
            metadata=metadata,
        )
        compiled.append(compiled_entry)

        planning_state = _apply_impulse_to_state(event_state, dv_vec)

    return tuple(compiled)


def simulate_scenario_series(
    propagator: Any,
    scenario: ScenarioStateFile,
    epoch_spec: OutputEpochSpec,
    *,
    series_name: str = "trajectory",
    representation: str = "cartesian",
    frame: str = "GCRF",
    mu_m3_s2: float | str = "WGS84",
    interpolation_samples: int = 8,
    universe: Mapping[str, Any] | None = None,
    default_mass_kg: float = 1000.0,
) -> tuple[StateSeries, tuple[CompiledManeuver, ...]]:
    """Propagate a trajectory while applying scenario maneuvers, then sample it."""
    if not isinstance(epoch_spec, OutputEpochSpec):
        raise TypeError("epoch_spec must be an OutputEpochSpec.")

    epochs = epoch_spec.epochs()
    if not epochs:
        raise ValueError("epoch_spec produced no epochs.")
    if not hasattr(propagator, "propagate"):
        raise TypeError("propagator must expose propagate(date).")
    if scenario.maneuvers and not hasattr(propagator, "resetInitialState"):
        raise TypeError(
            "propagator must expose resetInitialState(state) to execute scenario maneuvers."
        )

    rep = representation.strip().lower()
    if rep not in {"cartesian", "keplerian", "equinoctial"}:
        raise ValueError("representation must be one of {'cartesian', 'keplerian', 'equinoctial'}.")

    # Keep output ordering as requested, but execute propagation in chronological order.
    sample_pairs = [(epoch, to_orekit_date(epoch)) for epoch in epochs]
    anchor_date = sample_pairs[0][1]
    sample_pairs.sort(key=lambda item: float(item[1].durationFrom(anchor_date)))
    start_date = sample_pairs[0][1]
    start_state = propagator.propagate(start_date)

    compiled = compile_scenario_maneuvers(scenario, start_state) if scenario.maneuvers else ()
    compiled_dates = [to_orekit_date(item.epoch) for item in compiled]

    output_frame = resolve_frame(frame, universe=universe)
    records_by_epoch: dict[str, OrbitStateRecord] = {}
    maneuver_idx = 0

    for epoch, sample_date in sample_pairs:
        while maneuver_idx < len(compiled):
            event_date = compiled_dates[maneuver_idx]
            if float(sample_date.durationFrom(event_date)) < 0.0:
                break

            pre_event_state = propagator.propagate(event_date)
            maneuver = compiled[maneuver_idx]
            dv_vec = _tuple_to_vector(maneuver.dv_inertial_mps)
            post_event_state = _apply_impulse_to_state(pre_event_state, dv_vec)
            propagator.resetInitialState(post_event_state)
            maneuver_idx += 1

        state = propagator.propagate(sample_date)
        records_by_epoch[epoch] = _state_to_record(
            state,
            epoch=epoch,
            representation=rep,
            frame_name=frame,
            output_frame=output_frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=default_mass_kg,
        )

    ordered_records = tuple(records_by_epoch[epoch] for epoch in epochs)
    series = StateSeries(
        name=series_name,
        states=ordered_records,
        interpolation={"method": "scenario_maneuvers", "samples": int(interpolation_samples)},
    )
    return series, compiled


def export_scenario_series(
    propagator: Any,
    scenario: ScenarioStateFile,
    epoch_spec: OutputEpochSpec,
    output_path: str | Path,
    *,
    series_name: str = "trajectory",
    representation: str = "cartesian",
    frame: str = "GCRF",
    mu_m3_s2: float | str = "WGS84",
    interpolation_samples: int = 8,
    dense_yaml: bool = True,
    universe: Mapping[str, Any] | None = None,
    default_mass_kg: float = 1000.0,
) -> tuple[Path, tuple[CompiledManeuver, ...]]:
    """Execute scenario maneuvers during propagation and export sampled states."""
    series, compiled = simulate_scenario_series(
        propagator,
        scenario,
        epoch_spec,
        series_name=series_name,
        representation=representation,
        frame=frame,
        mu_m3_s2=mu_m3_s2,
        interpolation_samples=interpolation_samples,
        universe=universe,
        default_mass_kg=default_mass_kg,
    )

    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        saved = save_state_series_hdf5(path, series)
    else:
        saved = save_state_series_compact_with_style(path, series, dense_rows=dense_yaml)
    return saved, compiled


def _resolve_trigger_date(trigger: Mapping[str, Any], state: Any) -> tuple[str, Any]:
    if not isinstance(trigger, Mapping):
        raise TypeError("maneuver.trigger must be a mapping.")

    trigger_type = str(trigger.get("type", "epoch")).strip().lower()
    if trigger_type == "epoch":
        epoch = trigger.get("epoch")
        if not isinstance(epoch, str):
            raise ValueError("epoch trigger requires string field trigger.epoch.")
        date = to_orekit_date(epoch)
        if float(date.durationFrom(state.getDate())) < 0.0:
            raise ValueError(
                f"epoch trigger '{epoch}' is before current mission planning state epoch "
                f"{from_orekit_date(state.getDate())}."
            )
        return trigger_type, date

    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(state.getOrbit())
    n = float(kep.getKeplerianMeanMotion())
    m_now = _normalize_angle(float(kep.getMeanAnomaly()))

    if trigger_type == "perigee":
        if float(kep.getE()) < 1.0e-10:
            raise ValueError("perigee trigger requires non-circular orbit (e > 0).")
        dt = _delta_time_to_target_mean_anomaly(m_now, 0.0, n)
        return trigger_type, state.getDate().shiftedBy(dt)

    if trigger_type == "apogee":
        if float(kep.getE()) < 1.0e-10:
            raise ValueError("apogee trigger requires non-circular orbit (e > 0).")
        dt = _delta_time_to_target_mean_anomaly(m_now, math.pi, n)
        return trigger_type, state.getDate().shiftedBy(dt)

    if trigger_type in {"ascending_node", "descending_node"}:
        inc = float(kep.getI())
        if abs(math.sin(inc)) < 1.0e-10:
            raise ValueError("node trigger requires non-equatorial orbit (sin(i) != 0).")
        omega = float(kep.getPerigeeArgument())
        true_target = -omega if trigger_type == "ascending_node" else (math.pi - omega)
        m_target = _true_to_mean_anomaly(true_target, float(kep.getE()))
        dt = _delta_time_to_target_mean_anomaly(m_now, m_target, n)
        return trigger_type, state.getDate().shiftedBy(dt)

    raise ValueError(
        "Unsupported maneuver trigger type. "
        "Supported: {'epoch', 'perigee', 'apogee', 'ascending_node', 'descending_node'}."
    )


def _resolve_maneuver_trigger(
    trigger: Mapping[str, Any],
    state: Any,
    timeline: Mapping[str, ResolvedTimelineEvent],
) -> tuple[str, Any, dict[str, Any]]:
    trigger_type = str(trigger.get("type", "epoch")).strip().lower()
    if trigger_type != "event":
        resolved_type, date = _resolve_trigger_date(trigger, state)
        return resolved_type, date, {}

    event_id = str(trigger.get("event", "")).strip()
    if not event_id:
        raise ValueError("event trigger requires non-empty trigger.event reference.")
    if event_id not in timeline:
        raise ValueError(f"event trigger references unknown timeline event '{event_id}'.")

    resolved = timeline[event_id]
    date = to_orekit_date(resolved.epoch)
    if float(date.durationFrom(state.getDate())) < 0.0:
        raise ValueError(
            f"event '{event_id}' ({resolved.epoch}) is before current maneuver planning epoch "
            f"{from_orekit_date(state.getDate())}."
        )
    return resolved.event_type, date, {"timeline_event": event_id}


def _resolve_timeline_events(
    timeline: Sequence[TimelineEventRecord],
    initial_state: Any,
) -> dict[str, ResolvedTimelineEvent]:
    resolved: dict[str, ResolvedTimelineEvent] = {}
    event_states: dict[str, Any] = {}
    initial_epoch = from_orekit_date(initial_state.getDate())

    for item in timeline:
        point = dict(item.point)
        point_type = str(point.get("type", "")).strip().lower()
        if not point_type:
            raise ValueError(f"timeline event '{item.id}' requires point.type.")

        if point_type == "epoch":
            epoch_raw = point.get("epoch")
            if not isinstance(epoch_raw, str):
                raise ValueError(f"timeline event '{item.id}' with type=epoch requires point.epoch string.")
            date = to_orekit_date(epoch_raw)
            if float(date.durationFrom(initial_state.getDate())) < 0.0:
                raise ValueError(
                    f"timeline event '{item.id}' epoch {epoch_raw} is before mission start {initial_epoch}."
                )
            state = _keplerian_propagate_state(initial_state, date)
            resolved[item.id] = ResolvedTimelineEvent(id=item.id, event_type="epoch", epoch=from_orekit_date(date))
            event_states[item.id] = state
            continue

        if point_type == "elapsed":
            ref_id = str(point.get("from", "")).strip()
            if not ref_id:
                raise ValueError(f"timeline event '{item.id}' with type=elapsed requires point.from.")
            if ref_id not in resolved:
                raise ValueError(f"timeline event '{item.id}' references unknown/forward event '{ref_id}'.")
            dt_s = _parse_duration_seconds(point.get("dt"), key_name=f"timeline[{item.id}].point.dt")
            date = to_orekit_date(resolved[ref_id].epoch).shiftedBy(dt_s)
            state = _keplerian_propagate_state(initial_state, date)
            resolved[item.id] = ResolvedTimelineEvent(id=item.id, event_type="elapsed", epoch=from_orekit_date(date))
            event_states[item.id] = state
            continue

        if point_type in {"apogee", "perigee", "ascending_node", "descending_node"}:
            after_id = str(point.get("after", "")).strip()
            if after_id:
                if after_id not in event_states:
                    raise ValueError(f"timeline event '{item.id}' references unknown/forward event '{after_id}'.")
                base_state = event_states[after_id]
            else:
                base_state = initial_state

            occurrence = str(point.get("occurrence", "first")).strip().lower()
            n = int(point.get("n", 1))
            if n < 1:
                raise ValueError(f"timeline event '{item.id}' requires n >= 1 when provided.")

            if occurrence in {"first", "next"}:
                count = 1
            elif occurrence == "nth":
                count = n
            else:
                raise ValueError(
                    f"timeline event '{item.id}' unsupported occurrence '{occurrence}'. "
                    "Supported: {'first', 'next', 'nth'}."
                )

            current_state = base_state
            date = None
            for _ in range(count):
                _, date = _resolve_trigger_date({"type": point_type}, current_state)
                current_state = _keplerian_propagate_state(current_state, date)
            assert date is not None

            resolved[item.id] = ResolvedTimelineEvent(
                id=item.id,
                event_type=point_type,
                epoch=from_orekit_date(date),
            )
            event_states[item.id] = current_state
            continue

        raise ValueError(
            f"timeline event '{item.id}' has unsupported point.type '{point_type}'. "
            "Supported: {'epoch', 'elapsed', 'apogee', 'perigee', 'ascending_node', 'descending_node'}."
        )

    return resolved


def _resolve_delta_v_vector(model: Mapping[str, Any], state: Any, trigger_type: str):
    if not isinstance(model, Mapping):
        raise TypeError("maneuver.model must be a mapping.")

    model_type = str(model.get("type", "impulsive")).strip().lower()
    if model_type == "impulsive":
        dv_values = _to_vector_tuple(model.get("dv_mps"), key_name="model.dv_mps")
        frame_name = str(model.get("frame", "TNW")).strip().upper()
        return _local_to_inertial_delta_v(state, dv_values, frame_name)

    if model_type in {
        "intent",
        "raise_perigee",
        "raise_semimajor_axis",
        "maintain_semimajor_axis_above",
        "change_inclination",
    }:
        intent = str(model.get("intent", model_type)).strip().lower()
        if intent == "raise_perigee":
            if trigger_type != "apogee":
                raise ValueError("intent 'raise_perigee' currently requires trigger.type='apogee'.")
            return _intent_raise_perigee(model, state)
        if intent in {"raise_semimajor_axis", "maintain_semimajor_axis_above"}:
            return _intent_raise_semimajor_axis(model, state)
        if intent == "change_inclination":
            if trigger_type not in {"ascending_node", "descending_node", "epoch"}:
                raise ValueError(
                    "intent 'change_inclination' requires trigger.type in "
                    "{'ascending_node', 'descending_node', 'epoch'}."
                )
            return _intent_change_inclination(model, state)
        raise ValueError(
            "Unsupported intent maneuver. Supported intents: "
            "{'raise_perigee', 'raise_semimajor_axis', 'maintain_semimajor_axis_above', 'change_inclination'}."
        )

    raise ValueError("Unsupported maneuver model.type. Supported: {'impulsive', 'intent'}.")


def _intent_raise_perigee(model: Mapping[str, Any], state: Any):
    from org.orekit.orbits import KeplerianOrbit

    target_defined = "target_perigee_m" in model
    delta_defined = "delta_perigee_m" in model
    if target_defined == delta_defined:
        raise ValueError(
            "intent 'raise_perigee' requires exactly one of model.target_perigee_m or model.delta_perigee_m."
        )

    kep = KeplerianOrbit(state.getOrbit())
    mu = float(kep.getMu())
    a = float(kep.getA())
    e = float(kep.getE())
    ra = a * (1.0 + e)
    rp = a * (1.0 - e)
    target_rp = float(model["target_perigee_m"]) if target_defined else (rp + float(model["delta_perigee_m"]))

    if target_rp <= 0.0:
        raise ValueError("model.target_perigee_m must be positive.")
    if target_rp <= rp:
        raise ValueError("model.target_perigee_m must be greater than current perigee for raise_perigee intent.")
    if target_rp > ra:
        raise ValueError("model.target_perigee_m cannot exceed current apogee for a single apogee burn.")

    v_current = math.sqrt(mu * ((2.0 / ra) - (1.0 / a)))
    a_target = 0.5 * (ra + target_rp)
    v_target = math.sqrt(mu * ((2.0 / ra) - (1.0 / a_target)))
    delta_t = v_target - v_current
    return _local_to_inertial_delta_v(state, (delta_t, 0.0, 0.0), "TNW")


def _intent_raise_semimajor_axis(model: Mapping[str, Any], state: Any):
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(state.getOrbit())
    a_now = float(kep.getA())
    mu = float(kep.getMu())

    min_a = model.get("min_a_m")
    if min_a is not None and a_now >= float(min_a):
        return Vector3D(0.0, 0.0, 0.0)

    target_defined = "target_a_m" in model
    delta_defined = "delta_a_m" in model
    if target_defined == delta_defined:
        raise ValueError(
            "intent 'raise_semimajor_axis' requires exactly one of model.target_a_m or model.delta_a_m."
        )

    target_a = float(model["target_a_m"]) if target_defined else (a_now + float(model["delta_a_m"]))
    if target_a <= a_now:
        raise ValueError("target semimajor axis must be greater than current semimajor axis for raise intent.")

    pv = state.getPVCoordinates()
    r_norm = float(pv.getPosition().getNorm())
    v_now = float(pv.getVelocity().getNorm())
    inside = mu * ((2.0 / r_norm) - (1.0 / target_a))
    if inside <= 0.0:
        raise ValueError("Requested target semimajor axis is not reachable at this position.")
    v_target = math.sqrt(inside)
    delta_t = v_target - v_now
    return _local_to_inertial_delta_v(state, (delta_t, 0.0, 0.0), "TNW")


def _intent_change_inclination(model: Mapping[str, Any], state: Any):
    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(state.getOrbit())
    i_now_deg = math.degrees(float(kep.getI()))

    if "target_i_deg" in model:
        delta_i_deg = float(model["target_i_deg"]) - i_now_deg
    elif "delta_i_deg" in model:
        delta_i_deg = float(model["delta_i_deg"])
    else:
        raise ValueError("intent 'change_inclination' requires model.target_i_deg or model.delta_i_deg.")

    if abs(delta_i_deg) < 1.0e-12:
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        return Vector3D(0.0, 0.0, 0.0)

    pv = state.getPVCoordinates()
    pos = pv.getPosition()
    vel = pv.getVelocity()
    axis = _unit(pos)
    rotated = _rotate_vector_about_axis(vel, axis, math.radians(delta_i_deg))
    return rotated.add(vel.scalarMultiply(-1.0))


def _keplerian_propagate_state(state: Any, target_date: Any):
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.analytical import KeplerianPropagator

    if float(target_date.durationFrom(state.getDate())) < 0.0:
        raise ValueError("target_date must be >= current state date for maneuver planning.")

    kp = KeplerianPropagator(state.getOrbit())
    propagated = kp.propagate(target_date)
    return SpacecraftState(propagated.getOrbit(), float(state.getMass()))


def _apply_impulse_to_state(state: Any, delta_v: Any):
    from org.orekit.orbits import CartesianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import PVCoordinates

    orbit = state.getOrbit()
    frame = orbit.getFrame()
    pv = state.getPVCoordinates(frame)
    pos = pv.getPosition()
    vel = pv.getVelocity().add(delta_v)
    new_pv = PVCoordinates(pos, vel)
    new_orbit = CartesianOrbit(new_pv, frame, state.getDate(), float(orbit.getMu()))
    return SpacecraftState(new_orbit, float(state.getMass()))


def _local_to_inertial_delta_v(state: Any, components: tuple[float, float, float], frame_name: str):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    c1, c2, c3 = components
    basis = _local_basis_vectors(state, frame_name)
    b1, b2, b3 = basis
    return (
        b1.scalarMultiply(c1)
        .add(b2.scalarMultiply(c2))
        .add(b3.scalarMultiply(c3))
    )


def _local_basis_vectors(state: Any, frame_name: str):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    pv = state.getPVCoordinates()
    r = pv.getPosition()
    v = pv.getVelocity()
    w = _unit(Vector3D.crossProduct(r, v))

    if frame_name == "TNW":
        t = _unit(v)
        n = _unit(Vector3D.crossProduct(w, t))
        return t, n, w

    if frame_name == "RTN":
        r_hat = _unit(r)
        t_hat = _unit(Vector3D.crossProduct(w, r_hat))
        return r_hat, t_hat, w

    if frame_name == "INERTIAL":
        from org.hipparchus.geometry.euclidean.threed import Vector3D as V3

        return V3(1.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), V3(0.0, 0.0, 1.0)

    raise ValueError("Unsupported maneuver frame. Supported: {'TNW', 'RTN', 'INERTIAL'}.")


def _rotate_vector_about_axis(vector: Any, axis: Any, angle_rad: float):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    term1 = vector.scalarMultiply(c)
    term2 = Vector3D.crossProduct(axis, vector).scalarMultiply(s)
    term3 = axis.scalarMultiply(float(Vector3D.dotProduct(axis, vector)) * (1.0 - c))
    return term1.add(term2).add(term3)


def _delta_time_to_target_mean_anomaly(m_current: float, m_target: float, mean_motion: float) -> float:
    delta_m = _normalize_angle(m_target - m_current)
    if delta_m < 1.0e-12:
        delta_m = 2.0 * math.pi
    return delta_m / mean_motion


def _true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float:
    nu = float(true_anomaly)
    e = float(eccentricity)
    sin_half = math.sin(0.5 * nu)
    cos_half = math.cos(0.5 * nu)
    e_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - e) * sin_half,
        math.sqrt(1.0 + e) * cos_half,
    )
    mean_anomaly = e_anomaly - e * math.sin(e_anomaly)
    return _normalize_angle(mean_anomaly)


def _normalize_angle(angle_rad: float) -> float:
    twopi = 2.0 * math.pi
    value = math.fmod(angle_rad, twopi)
    if value < 0.0:
        value += twopi
    return value


def _unit(vector: Any):
    norm = float(vector.getNorm())
    if norm <= 0.0:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector.scalarMultiply(1.0 / norm)


def _to_vector_tuple(values: Any, *, key_name: str) -> tuple[float, float, float]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{key_name} must be a 3-element numeric sequence.")
    if len(values) != 3:
        raise ValueError(f"{key_name} must contain exactly 3 components.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _parse_duration_seconds(value: Any, *, key_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"{key_name} must be numeric seconds or duration string.")

    text = value.strip().lower()
    if not text:
        raise ValueError(f"{key_name} cannot be empty.")
    if text.endswith("h"):
        return float(text[:-1]) * 3600.0
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    if text.endswith("s"):
        return float(text[:-1])
    return float(text)


def _tuple_to_vector(values: Sequence[float]):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    return Vector3D(float(values[0]), float(values[1]), float(values[2]))


def _state_to_record(
    state: Any,
    *,
    epoch: str,
    representation: str,
    frame_name: str,
    output_frame: Any,
    mu_m3_s2: float | str,
    default_mass_kg: float,
) -> OrbitStateRecord:
    from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit

    mass = float(state.getMass()) if hasattr(state, "getMass") else float(default_mass_kg)
    orbit = state.getOrbit()
    mu = orbit.getMu()

    if representation == "cartesian":
        pv = state.getPVCoordinates(output_frame)
        pos = pv.getPosition()
        vel = pv.getVelocity()
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame_name,
            representation="cartesian",
            position_m=(pos.getX(), pos.getY(), pos.getZ()),
            velocity_mps=(vel.getX(), vel.getY(), vel.getZ()),
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    orbit_in_frame = orbit
    if orbit.getFrame() != output_frame:
        pv = state.getPVCoordinates(output_frame)
        orbit_in_frame = CartesianOrbit(pv, output_frame, state.getDate(), mu)

    if representation == "keplerian":
        kep = KeplerianOrbit(orbit_in_frame)
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame_name,
            representation="keplerian",
            elements={
                "a_m": float(kep.getA()),
                "e": float(kep.getE()),
                "i_deg": math.degrees(float(kep.getI())),
                "argp_deg": math.degrees(float(kep.getPerigeeArgument())),
                "raan_deg": math.degrees(float(kep.getRightAscensionOfAscendingNode())),
                "anomaly_deg": math.degrees(float(kep.getMeanAnomaly())),
                "anomaly_type": "MEAN",
            },
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    equi = EquinoctialOrbit(orbit_in_frame)
    return OrbitStateRecord(
        epoch=epoch,
        frame=frame_name,
        representation="equinoctial",
        elements={
            "a_m": float(equi.getA()),
            "ex": float(equi.getEquinoctialEx()),
            "ey": float(equi.getEquinoctialEy()),
            "hx": float(equi.getHx()),
            "hy": float(equi.getHy()),
            "l_deg": math.degrees(float(equi.getLM())),
            "anomaly_type": "MEAN",
        },
        mu_m3_s2=mu_m3_s2,
        mass_kg=mass,
    )
