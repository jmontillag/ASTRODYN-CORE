"""Orekit EventDetector factory for scenario-driven maneuver triggers.

Each maneuver in a scenario is translated to an Orekit EventDetector that
fires when the physical trigger condition is met during numerical propagation.
The attached Python EventHandler evaluates optional guard conditions and applies
the impulse via ``resetState()``.

Supported trigger types
-----------------------
- ``apogee`` / ``perigee`` → ``ApsideDetector``
- ``ascending_node`` / ``descending_node`` → ``NodeDetector``
- ``epoch`` / ``elapsed`` / ``event`` → ``DateDetector``

Optional trigger dict keys (do not affect existing maneuver files)
------------------------------------------------------------------
- ``occurrence``: ``"first"`` | ``"every"`` (default) | ``"nth"`` | ``"limited"``
- ``n``: integer, used with ``"nth"``
- ``max``: integer, used with ``"limited"``
- ``guard``: dict with optional orbital guard conditions
- ``active_window``: dict with optional ``start`` / ``end`` epoch strings
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableSequence, Sequence

from astrodyn_core.states.models import ManeuverRecord, ScenarioStateFile, TimelineEventRecord
from astrodyn_core.states.orekit import from_orekit_date, to_orekit_date


# ---------------------------------------------------------------------------
# Guard evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_guard(state: Any, guard: Mapping[str, Any]) -> str | None:
    """Evaluate guard conditions.  Returns a skip-reason string or None if pass."""
    if not guard:
        return None

    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(state.getOrbit())
    a = float(kep.getA())
    e = float(kep.getE())
    re = 6_378_137.0  # Earth equatorial radius, m

    if "sma_above_m" in guard and a >= float(guard["sma_above_m"]):
        return f"sma {a:.0f} m >= sma_above_m {guard['sma_above_m']} m"

    if "sma_below_m" in guard and a <= float(guard["sma_below_m"]):
        return f"sma {a:.0f} m <= sma_below_m {guard['sma_below_m']} m"

    # Approximate altitude from current SMA and eccentricity at the event point
    # (exact location depends on trigger, so we use current radius as proxy)
    pv = state.getPVCoordinates()
    r = float(pv.getPosition().getNorm())
    alt = r - re

    if "altitude_above_m" in guard and alt >= float(guard["altitude_above_m"]):
        return (
            f"altitude {alt:.0f} m >= altitude_above_m {guard['altitude_above_m']} m"
        )

    if "altitude_below_m" in guard and alt <= float(guard["altitude_below_m"]):
        return (
            f"altitude {alt:.0f} m <= altitude_below_m {guard['altitude_below_m']} m"
        )

    return None


def _check_active_window(
    state: Any,
    active_window: Mapping[str, Any] | None,
) -> str | None:
    """Return a skip-reason string if the event is outside its active window."""
    if not active_window:
        return None

    epoch = from_orekit_date(state.getDate())

    start_raw = active_window.get("start")
    end_raw = active_window.get("end")

    if start_raw is not None:
        start_date = to_orekit_date(str(start_raw))
        if float(state.getDate().durationFrom(start_date)) < 0.0:
            return f"event time {epoch} is before active_window.start {start_raw}"

    if end_raw is not None:
        end_date = to_orekit_date(str(end_raw))
        if float(state.getDate().durationFrom(end_date)) > 0.0:
            return f"event time {epoch} is after active_window.end {end_raw}"

    return None


# ---------------------------------------------------------------------------
# Occurrence policy
# ---------------------------------------------------------------------------

class _OccurrencePolicy:
    """Tracks occurrence count and determines whether to apply a maneuver."""

    def __init__(self, occurrence: str, n: int, max_count: int) -> None:
        self.occurrence = occurrence
        self.target_n = n
        self.max_count = max_count
        self.count = 0

    def should_apply(self) -> bool:
        """Call *before* incrementing count. Returns True if the maneuver should fire."""
        self.count += 1
        if self.occurrence == "first":
            return self.count == 1
        if self.occurrence == "every":
            return self.max_count <= 0 or self.count <= self.max_count
        if self.occurrence == "nth":
            return self.count == self.target_n
        if self.occurrence == "limited":
            return self.count <= self.max_count
        return True

    @classmethod
    def from_trigger(cls, trigger: Mapping[str, Any]) -> _OccurrencePolicy:
        occurrence = str(trigger.get("occurrence", "every")).strip().lower()
        n = int(trigger.get("n", 1))
        max_count = int(trigger.get("max", 0))
        if occurrence not in {"first", "every", "nth", "limited"}:
            raise ValueError(
                f"Unknown trigger.occurrence {occurrence!r}. "
                "Supported: {'first', 'every', 'nth', 'limited'}."
            )
        if occurrence == "nth" and n < 1:
            raise ValueError("trigger.n must be >= 1 for occurrence='nth'.")
        if occurrence == "limited" and max_count < 1:
            raise ValueError("trigger.max must be >= 1 for occurrence='limited'.")
        return cls(occurrence, n, max_count)


# ---------------------------------------------------------------------------
# Impulse application (re-used from maneuvers module)
# ---------------------------------------------------------------------------

def _apply_impulse(state: Any, dv_inertial: tuple[float, float, float]) -> Any:
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.orbits import CartesianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import PVCoordinates

    orbit = state.getOrbit()
    frame = orbit.getFrame()
    pv = state.getPVCoordinates(frame)
    pos = pv.getPosition()
    dv = Vector3D(float(dv_inertial[0]), float(dv_inertial[1]), float(dv_inertial[2]))
    vel = pv.getVelocity().add(dv)
    new_pv = PVCoordinates(pos, vel)
    new_orbit = CartesianOrbit(new_pv, frame, state.getDate(), float(orbit.getMu()))
    return SpacecraftState(new_orbit, float(state.getMass()))


def _resolve_dv(model: Mapping[str, Any], state: Any, trigger_type: str) -> Any:
    """Delegate to mission intent delta-v resolver at detector fire time."""
    from astrodyn_core.mission.intents import resolve_delta_v_vector

    return resolve_delta_v_vector(model, state, trigger_type)


# ---------------------------------------------------------------------------
# ManeuverEventHandler: Python implementation of Orekit EventHandler
# ---------------------------------------------------------------------------

def _make_maneuver_handler(
    maneuver_name: str,
    trigger_type: str,
    model: Mapping[str, Any],
    guard: Mapping[str, Any],
    active_window: Mapping[str, Any] | None,
    occurrence: _OccurrencePolicy,
    execution_log: MutableSequence[Any],
):
    """Return an Orekit-compatible Python EventHandler object.

    Uses the JPype interface-implementation pattern available in the Orekit
    Python wrapper to subclass the Java EventHandler interface.
    """
    from org.hipparchus.ode.events import Action
    from org.orekit.propagation.events.handlers import PythonEventHandler as EventHandler

    class _ManeuverHandler(EventHandler):
        """Applies a single scenario maneuver impulse when the detector fires."""

        def eventOccurred(self, state, detector, increasing):
            # Check active window
            window_skip = _check_active_window(state, active_window)
            if window_skip is not None:
                _append_log(execution_log, maneuver_name, trigger_type, state,
                            applied=False, skip_reason=f"active_window: {window_skip}",
                            dv=None)
                return Action.CONTINUE

            # Check occurrence policy
            if not occurrence.should_apply():
                _append_log(execution_log, maneuver_name, trigger_type, state,
                            applied=False, skip_reason=f"occurrence policy skipped",
                            dv=None)
                return Action.CONTINUE

            # Check guard conditions
            guard_skip = _evaluate_guard(state, guard)
            if guard_skip is not None:
                _append_log(execution_log, maneuver_name, trigger_type, state,
                            applied=False, skip_reason=f"guard: {guard_skip}",
                            dv=None)
                return Action.CONTINUE

            # Resolve delta-v at event time
            try:
                dv_vec = _resolve_dv(model, state, trigger_type)
            except Exception as exc:
                _append_log(execution_log, maneuver_name, trigger_type, state,
                            applied=False, skip_reason=f"dv resolution error: {exc}",
                            dv=None)
                return Action.CONTINUE

            dv_tuple = (float(dv_vec.getX()), float(dv_vec.getY()), float(dv_vec.getZ()))
            dv_norm = math.sqrt(sum(c * c for c in dv_tuple))

            if dv_norm < 1e-12:
                # Zero delta-v (e.g., guard intent already satisfied): no state reset needed
                _append_log(execution_log, maneuver_name, trigger_type, state,
                            applied=False, skip_reason="zero delta-v (intent already satisfied)",
                            dv=dv_tuple)
                return Action.CONTINUE

            _append_log(execution_log, maneuver_name, trigger_type, state,
                        applied=True, skip_reason=None, dv=dv_tuple)
            return Action.RESET_STATE

        def init(self, s0, t, detector):
            pass

        def resetState(self, detector, old_state):
            # Recompute dv at old_state (same as eventOccurred) and apply
            try:
                dv_vec = _resolve_dv(model, old_state, trigger_type)
            except Exception:
                return old_state
            dv_tuple = (float(dv_vec.getX()), float(dv_vec.getY()), float(dv_vec.getZ()))
            return _apply_impulse(old_state, dv_tuple)

        def finish(self, final_state, detector):
            pass

    return _ManeuverHandler()


def _append_log(
    log: MutableSequence[Any],
    name: str,
    trigger_type: str,
    state: Any,
    *,
    applied: bool,
    skip_reason: str | None,
    dv: tuple[float, float, float] | None,
) -> None:
    """Append a log entry dict to the execution log (converted to dataclass later)."""
    log.append({
        "maneuver_name": name,
        "trigger_type": trigger_type,
        "epoch": from_orekit_date(state.getDate()),
        "applied": applied,
        "guard_skip_reason": skip_reason,
        "dv_inertial_mps": dv,
    })


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def _build_apside_detector(orbit: Any, trigger_type: str, handler: Any) -> Any:
    from org.orekit.propagation.events import ApsideDetector

    detector = ApsideDetector(orbit)
    return detector.withHandler(handler)


def _build_node_detector(orbit: Any, trigger_type: str, handler: Any) -> Any:
    from org.orekit.propagation.events import NodeDetector

    frame = orbit.getFrame()
    detector = NodeDetector(orbit, frame)
    return detector.withHandler(handler)


def _build_date_detector(epoch_str: str, handler: Any) -> Any:
    from org.orekit.propagation.events import DateDetector

    date = to_orekit_date(epoch_str)
    detector = DateDetector(date)
    return detector.withHandler(handler)


def _resolved_epoch_for_event_trigger(
    trigger: Mapping[str, Any],
    initial_state: Any,
    resolved_timeline: Mapping[str, Any],
) -> str:
    """Return the epoch string for an event-reference trigger."""
    trigger_type = str(trigger.get("type", "epoch")).strip().lower()

    if trigger_type == "epoch":
        return str(trigger["epoch"])

    if trigger_type == "elapsed":
        from astrodyn_core.mission.timeline import parse_duration_seconds
        from_epoch = str(trigger.get("from_epoch", ""))
        dt_s = parse_duration_seconds(trigger.get("dt"), key_name="trigger.dt")
        base_date = to_orekit_date(from_epoch) if from_epoch else initial_state.getDate()
        return from_orekit_date(base_date.shiftedBy(dt_s))

    if trigger_type == "event":
        event_id = str(trigger.get("event", "")).strip()
        if event_id not in resolved_timeline:
            raise ValueError(
                f"trigger.event references unknown timeline event '{event_id}'."
            )
        return resolved_timeline[event_id].epoch

    raise ValueError(
        f"trigger.type={trigger_type!r} is not a date-based trigger. "
        "Use apogee/perigee/ascending_node/descending_node for orbital triggers."
    )


def build_detectors_from_scenario(
    scenario: ScenarioStateFile,
    initial_state: Any,
    resolved_timeline: Mapping[str, Any],
    execution_log: MutableSequence[Any],
    universe: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Build Orekit EventDetector instances for all maneuvers in a scenario.

    Parameters
    ----------
    scenario:
        The scenario file. ``scenario.maneuvers`` is iterated.
    initial_state:
        Orekit ``SpacecraftState`` at the mission start epoch.
    resolved_timeline:
        Mapping ``{event_id: ResolvedTimelineEvent}`` as returned by
        ``mission.timeline.resolve_timeline_events``.
    execution_log:
        Mutable list that each handler appends log-entry dicts to during
        propagation. Converted to ``ManeuverFiredEvent`` objects by the executor.
    universe:
        Optional universe configuration (passed through for frame resolution).

    Returns
    -------
    list
        List of Orekit ``EventDetector`` instances.
    """
    orbit = initial_state.getOrbit()
    detectors: list[Any] = []

    for maneuver in scenario.maneuvers:
        trigger = dict(maneuver.trigger)
        model = dict(maneuver.model)
        trigger_type = str(trigger.get("type", "epoch")).strip().lower()

        guard = dict(trigger.get("guard", {}))
        active_window = trigger.get("active_window")
        occurrence = _OccurrencePolicy.from_trigger(trigger)

        handler = _make_maneuver_handler(
            maneuver_name=maneuver.name,
            trigger_type=trigger_type,
            model=model,
            guard=guard,
            active_window=active_window,
            occurrence=occurrence,
            execution_log=execution_log,
        )

        if trigger_type == "apogee":
            det = _build_apside_detector(orbit, trigger_type, handler)
        elif trigger_type == "perigee":
            det = _build_apside_detector(orbit, trigger_type, handler)
        elif trigger_type in {"ascending_node", "descending_node"}:
            det = _build_node_detector(orbit, trigger_type, handler)
        elif trigger_type in {"epoch", "elapsed", "event"}:
            epoch_str = _resolved_epoch_for_event_trigger(
                trigger, initial_state, resolved_timeline
            )
            det = _build_date_detector(epoch_str, handler)
        else:
            raise ValueError(
                f"Maneuver '{maneuver.name}' has unsupported trigger.type={trigger_type!r}. "
                "Supported: apogee, perigee, ascending_node, descending_node, epoch, elapsed, event."
            )

        detectors.append(det)

    return detectors
