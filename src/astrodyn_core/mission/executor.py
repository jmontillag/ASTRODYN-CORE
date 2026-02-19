"""Detector-driven scenario executor (closed-loop maneuver execution).

This module provides :class:`ScenarioExecutor`, which binds maneuver triggers
from a scenario file to Orekit ``EventDetector`` instances and propagates the
full mission in closed-loop mode — meaning maneuvers fire precisely when the
physical trigger condition is met during numerical integration, rather than at
precomputed Keplerian-approximated times.

The execution report records every triggered event, whether it was applied or
skipped by a guard/occurrence/window condition, and the total delta-v consumed.

Relationship to the existing pipeline
--------------------------------------
- The existing :func:`~astrodyn_core.mission.maneuvers.simulate_scenario_series`
  remains available for rapid Keplerian-approximation-based design iteration.
- :class:`ScenarioExecutor` is the higher-fidelity alternative for validation
  and operational planning with numerical propagators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from astrodyn_core.states.models import (
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
)
from astrodyn_core.states.orekit import from_orekit_date, resolve_frame, to_orekit_date


# ---------------------------------------------------------------------------
# Execution report models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ManeuverFiredEvent:
    """Record of a single maneuver trigger event during detector-driven execution.

    Attributes
    ----------
    maneuver_name:
        Name of the maneuver as defined in the scenario file.
    epoch:
        ISO-8601 UTC string of the exact time the detector fired.
    trigger_type:
        The trigger condition that fired (``"apogee"``, ``"perigee"``, etc.).
    dv_inertial_mps:
        Applied delta-v vector in the inertial frame (m/s). ``None`` if skipped
        before delta-v resolution.
    applied:
        ``True`` if the impulse was applied; ``False`` if skipped by a guard,
        occurrence policy, or active-window constraint.
    guard_skip_reason:
        Human-readable reason for skipping, or ``None`` if the maneuver fired.
    """

    maneuver_name: str
    epoch: str
    trigger_type: str
    dv_inertial_mps: tuple[float, float, float] | None
    applied: bool
    guard_skip_reason: str | None


@dataclass(frozen=True, slots=True)
class MissionExecutionReport:
    """Summary of a detector-driven scenario execution.

    Attributes
    ----------
    events:
        Tuple of all :class:`ManeuverFiredEvent` instances, in chronological order.
    total_dv_mps:
        Scalar total delta-v applied (m/s), summed over all applied events.
    propagation_start:
        ISO-8601 UTC string for the propagation start epoch.
    propagation_end:
        ISO-8601 UTC string for the propagation end epoch.
    """

    events: tuple[ManeuverFiredEvent, ...]
    total_dv_mps: float
    propagation_start: str
    propagation_end: str

    def applied_events(self) -> tuple[ManeuverFiredEvent, ...]:
        """Return only events where the maneuver was actually applied."""
        return tuple(e for e in self.events if e.applied)

    def skipped_events(self) -> tuple[ManeuverFiredEvent, ...]:
        """Return only events that were skipped."""
        return tuple(e for e in self.events if not e.applied)


# ---------------------------------------------------------------------------
# ScenarioExecutor
# ---------------------------------------------------------------------------

class ScenarioExecutor:
    """Executes a scenario using Orekit event detectors for maneuver triggering.

    Usage::

        executor = ScenarioExecutor(propagator, scenario)
        state_series, report = executor.run_and_sample(epoch_spec)

    Parameters
    ----------
    propagator:
        A built Orekit NumericalPropagator (or DSST) instance. Event detectors
        will be added to it via ``addEventDetector()``. The propagator's initial
        state is used as the mission start epoch.
    scenario:
        Loaded :class:`~astrodyn_core.states.models.ScenarioStateFile`.
    universe:
        Optional universe configuration for frame resolution.
    """

    def __init__(
        self,
        propagator: Any,
        scenario: ScenarioStateFile,
        *,
        universe: Mapping[str, Any] | None = None,
    ) -> None:
        if not hasattr(propagator, "addEventDetector"):
            raise TypeError(
                "propagator must expose addEventDetector(). "
                "Use a NumericalPropagator, not a builder."
            )
        if not isinstance(scenario, ScenarioStateFile):
            raise TypeError("scenario must be a ScenarioStateFile instance.")

        self._propagator = propagator
        self._scenario = scenario
        self._universe = universe
        self._execution_log: list[dict] = []
        self._configured = False

    def configure(self) -> None:
        """Resolve the scenario timeline and register all detectors on the propagator.

        Called automatically by :meth:`run_and_sample` if not called explicitly.
        Calling it explicitly allows inspecting the detectors before propagation.
        """
        from astrodyn_core.mission.detectors import build_detectors_from_scenario
        from astrodyn_core.mission.maneuvers import _resolve_timeline_events

        self._execution_log.clear()
        initial_state = self._propagator.getInitialState()

        # Resolve timeline events with Keplerian approximation (for date-based detectors)
        resolved_timeline = _resolve_timeline_events(
            self._scenario.timeline, initial_state
        )

        detectors = build_detectors_from_scenario(
            self._scenario,
            initial_state,
            resolved_timeline,
            self._execution_log,
            universe=self._universe,
        )

        self._propagator.clearEventsDetectors()
        for det in detectors:
            self._propagator.addEventDetector(det)

        self._configured = True
        self._start_epoch = from_orekit_date(initial_state.getDate())

    def run(self, target_date_or_epoch: Any) -> tuple[Any, MissionExecutionReport]:
        """Propagate to the target epoch and return the final state + report.

        Parameters
        ----------
        target_date_or_epoch:
            Either an Orekit ``AbsoluteDate`` or an ISO-8601 UTC string.

        Returns
        -------
        tuple[SpacecraftState, MissionExecutionReport]
        """
        if not self._configured:
            self.configure()

        if isinstance(target_date_or_epoch, str):
            target_date = to_orekit_date(target_date_or_epoch)
        else:
            target_date = target_date_or_epoch

        end_state = self._propagator.propagate(target_date)
        report = self._build_report(from_orekit_date(target_date))
        return end_state, report

    def run_and_sample(
        self,
        epoch_spec: OutputEpochSpec,
        *,
        series_name: str = "trajectory",
        representation: str = "cartesian",
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        default_mass_kg: float = 1000.0,
    ) -> tuple[StateSeries, MissionExecutionReport]:
        """Propagate the full mission and sample the trajectory at output epochs.

        The propagator runs detector-driven to the last epoch in ``epoch_spec``,
        applying maneuvers when detectors fire. After propagation, a second pass
        samples the trajectory at the requested output epochs using the propagator's
        built-in ephemeris (the propagator is used in ``EPHEMERIS_GENERATION_MODE``
        internally via Orekit's step handling).

        For simplicity, this implementation propagates to each output epoch in
        sequence and records the state. Because Orekit's NumericalPropagator
        tracks state resets from detectors, each ``propagate(date)`` call returns
        the physically correct post-maneuver state.

        Parameters
        ----------
        epoch_spec:
            Output epoch specification.
        series_name:
            Name for the returned :class:`~astrodyn_core.states.models.StateSeries`.
        representation:
            Output orbit representation.
        frame:
            Output frame name.
        mu_m3_s2:
            Gravitational parameter for output records.
        default_mass_kg:
            Fallback spacecraft mass.

        Returns
        -------
        tuple[StateSeries, MissionExecutionReport]
        """
        if not isinstance(epoch_spec, OutputEpochSpec):
            raise TypeError("epoch_spec must be an OutputEpochSpec instance.")

        epochs = epoch_spec.epochs()
        if not epochs:
            raise ValueError("epoch_spec produced no epochs.")

        rep = representation.strip().lower()
        if rep not in {"cartesian", "keplerian", "equinoctial"}:
            raise ValueError(
                "representation must be one of {'cartesian', 'keplerian', 'equinoctial'}."
            )

        if not self._configured:
            self.configure()

        output_frame = resolve_frame(frame, universe=self._universe)

        # Sort epochs chronologically for propagation; restore original order for output.
        epoch_list = list(epochs)
        sorted_epochs = sorted(
            epoch_list,
            key=lambda ep: float(to_orekit_date(ep).durationFrom(to_orekit_date(epoch_list[0]))),
        )

        records_by_epoch: dict[str, OrbitStateRecord] = {}
        for epoch_str in sorted_epochs:
            target_date = to_orekit_date(epoch_str)
            state = self._propagator.propagate(target_date)
            records_by_epoch[epoch_str] = _state_to_record(
                state,
                epoch=epoch_str,
                representation=rep,
                frame_name=frame,
                output_frame=output_frame,
                mu_m3_s2=mu_m3_s2,
                default_mass_kg=default_mass_kg,
            )

        ordered_records = tuple(records_by_epoch[ep] for ep in epoch_list)
        state_series = StateSeries(
            name=series_name,
            states=ordered_records,
            interpolation={"method": "detector_driven"},
        )
        report = self._build_report(sorted_epochs[-1] if sorted_epochs else "")
        return state_series, report

    def _build_report(self, propagation_end: str) -> MissionExecutionReport:
        events = tuple(
            ManeuverFiredEvent(
                maneuver_name=entry["maneuver_name"],
                epoch=entry["epoch"],
                trigger_type=entry["trigger_type"],
                dv_inertial_mps=entry.get("dv_inertial_mps"),
                applied=bool(entry["applied"]),
                guard_skip_reason=entry.get("guard_skip_reason"),
            )
            for entry in self._execution_log
        )
        total_dv = sum(
            math.sqrt(sum(c * c for c in e.dv_inertial_mps))
            for e in events
            if e.applied and e.dv_inertial_mps is not None
        )
        return MissionExecutionReport(
            events=events,
            total_dv_mps=round(total_dv, 6),
            propagation_start=getattr(self, "_start_epoch", ""),
            propagation_end=propagation_end,
        )


# ---------------------------------------------------------------------------
# State-to-record conversion (mirrors maneuvers.py for consistency)
# ---------------------------------------------------------------------------

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
    import math

    from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit

    mass = float(state.getMass()) if hasattr(state, "getMass") else float(default_mass_kg)
    orbit = state.getOrbit()
    mu = float(orbit.getMu())

    if representation == "cartesian":
        pv = state.getPVCoordinates(output_frame)
        pos = pv.getPosition()
        vel = pv.getVelocity()
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame_name,
            representation="cartesian",
            position_m=(float(pos.getX()), float(pos.getY()), float(pos.getZ())),
            velocity_mps=(float(vel.getX()), float(vel.getY()), float(vel.getZ())),
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
