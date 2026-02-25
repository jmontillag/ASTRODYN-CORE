"""Timeline and trigger resolution helpers for mission workflows."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from astrodyn_core.mission.models import ResolvedTimelineEvent
from astrodyn_core.states.models import TimelineEventRecord
from astrodyn_core.states.orekit_dates import from_orekit_date, to_orekit_date


def resolve_trigger_date(trigger: Mapping[str, Any], state: Any) -> tuple[str, Any]:
    """Resolve a maneuver trigger to a concrete Orekit date.

    Args:
        trigger: Maneuver trigger mapping.
        state: Orekit ``SpacecraftState`` used as the planning reference state.

    Returns:
        Tuple ``(trigger_type, date)`` where ``date`` is an Orekit absolute date.

    Raises:
        TypeError: If ``trigger`` is not a mapping.
        ValueError: If the trigger is invalid or not reachable from ``state``.
    """
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
    m_now = normalize_angle(float(kep.getMeanAnomaly()))

    if trigger_type == "perigee":
        if float(kep.getE()) < 1.0e-10:
            raise ValueError("perigee trigger requires non-circular orbit (e > 0).")
        dt = delta_time_to_target_mean_anomaly(m_now, 0.0, n)
        return trigger_type, state.getDate().shiftedBy(dt)

    if trigger_type == "apogee":
        if float(kep.getE()) < 1.0e-10:
            raise ValueError("apogee trigger requires non-circular orbit (e > 0).")
        dt = delta_time_to_target_mean_anomaly(m_now, math.pi, n)
        return trigger_type, state.getDate().shiftedBy(dt)

    if trigger_type in {"ascending_node", "descending_node"}:
        inc = float(kep.getI())
        if abs(math.sin(inc)) < 1.0e-10:
            raise ValueError("node trigger requires non-equatorial orbit (sin(i) != 0).")
        omega = float(kep.getPerigeeArgument())
        true_target = -omega if trigger_type == "ascending_node" else (math.pi - omega)
        m_target = true_to_mean_anomaly(true_target, float(kep.getE()))
        dt = delta_time_to_target_mean_anomaly(m_now, m_target, n)
        return trigger_type, state.getDate().shiftedBy(dt)

    raise ValueError(
        "Unsupported maneuver trigger type. "
        "Supported: {'epoch', 'perigee', 'apogee', 'ascending_node', 'descending_node'}."
    )


def resolve_maneuver_trigger(
    trigger: Mapping[str, Any],
    state: Any,
    timeline: Mapping[str, ResolvedTimelineEvent],
) -> tuple[str, Any, dict[str, Any]]:
    """Resolve a maneuver trigger, including timeline event references.

    Args:
        trigger: Maneuver trigger mapping.
        state: Orekit planning state used as the reference epoch.
        timeline: Resolved timeline event mapping.

    Returns:
        Tuple ``(trigger_type, date, metadata)``.
    """
    trigger_type = str(trigger.get("type", "epoch")).strip().lower()
    if trigger_type != "event":
        resolved_type, date = resolve_trigger_date(trigger, state)
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


def resolve_timeline_events(
    timeline: Sequence[TimelineEventRecord],
    initial_state: Any,
) -> dict[str, ResolvedTimelineEvent]:
    """Resolve all timeline events to absolute epochs.

    Args:
        timeline: Timeline event records in scenario order.
        initial_state: Orekit mission-start state.

    Returns:
        Mapping from event id to resolved timeline event.

    Raises:
        ValueError: If timeline event definitions are invalid or reference
            unknown/forward events.
    """
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
                raise ValueError(
                    f"timeline event '{item.id}' with type=epoch requires point.epoch string."
                )
            date = to_orekit_date(epoch_raw)
            if float(date.durationFrom(initial_state.getDate())) < 0.0:
                raise ValueError(
                    f"timeline event '{item.id}' epoch {epoch_raw} is before mission start {initial_epoch}."
                )
            state = keplerian_propagate_state(initial_state, date)
            resolved[item.id] = ResolvedTimelineEvent(
                id=item.id, event_type="epoch", epoch=from_orekit_date(date)
            )
            event_states[item.id] = state
            continue

        if point_type == "elapsed":
            ref_id = str(point.get("from", "")).strip()
            if not ref_id:
                raise ValueError(
                    f"timeline event '{item.id}' with type=elapsed requires point.from."
                )
            if ref_id not in resolved:
                raise ValueError(
                    f"timeline event '{item.id}' references unknown/forward event '{ref_id}'."
                )
            dt_s = parse_duration_seconds(point.get("dt"), key_name=f"timeline[{item.id}].point.dt")
            date = to_orekit_date(resolved[ref_id].epoch).shiftedBy(dt_s)
            state = keplerian_propagate_state(initial_state, date)
            resolved[item.id] = ResolvedTimelineEvent(
                id=item.id, event_type="elapsed", epoch=from_orekit_date(date)
            )
            event_states[item.id] = state
            continue

        if point_type in {"apogee", "perigee", "ascending_node", "descending_node"}:
            after_id = str(point.get("after", "")).strip()
            if after_id:
                if after_id not in event_states:
                    raise ValueError(
                        f"timeline event '{item.id}' references unknown/forward event '{after_id}'."
                    )
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
                _, date = resolve_trigger_date({"type": point_type}, current_state)
                current_state = keplerian_propagate_state(current_state, date)
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


def keplerian_propagate_state(state: Any, target_date: Any):
    """Propagate a state with a temporary Keplerian propagator for planning."""
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.analytical import KeplerianPropagator

    if float(target_date.durationFrom(state.getDate())) < 0.0:
        raise ValueError("target_date must be >= current state date for maneuver planning.")

    kp = KeplerianPropagator(state.getOrbit())
    propagated = kp.propagate(target_date)
    return SpacecraftState(propagated.getOrbit(), float(state.getMass()))


def delta_time_to_target_mean_anomaly(
    m_current: float, m_target: float, mean_motion: float
) -> float:
    """Return positive time-to-go for the next target mean anomaly crossing."""
    delta_m = normalize_angle(m_target - m_current)
    if delta_m < 1.0e-12:
        delta_m = 2.0 * math.pi
    return delta_m / mean_motion


def true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly to mean anomaly for an elliptical orbit."""
    nu = float(true_anomaly)
    e = float(eccentricity)
    sin_half = math.sin(0.5 * nu)
    cos_half = math.cos(0.5 * nu)
    e_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - e) * sin_half,
        math.sqrt(1.0 + e) * cos_half,
    )
    mean_anomaly = e_anomaly - e * math.sin(e_anomaly)
    return normalize_angle(mean_anomaly)


def normalize_angle(angle_rad: float) -> float:
    """Wrap an angle into the ``[0, 2π)`` interval."""
    twopi = 2.0 * math.pi
    value = math.fmod(angle_rad, twopi)
    if value < 0.0:
        value += twopi
    return value


def parse_duration_seconds(value: Any, *, key_name: str) -> float:
    """Parse a numeric/compact duration value into seconds.

    Supports raw numeric seconds and strings with suffixes ``s``, ``m``, ``h``.

    Args:
        value: Duration value.
        key_name: Config key name used in error messages.

    Returns:
        Duration in seconds.
    """
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
