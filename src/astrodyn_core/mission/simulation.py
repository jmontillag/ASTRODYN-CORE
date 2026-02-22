"""Scenario maneuver compilation, simulation, and export workflows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.mission.intents import resolve_delta_v_vector
from astrodyn_core.mission.kinematics import tuple_to_vector
from astrodyn_core.mission.models import CompiledManeuver
from astrodyn_core.mission.timeline import resolve_maneuver_trigger, resolve_timeline_events
from astrodyn_core.states.io import save_state_series_compact_with_style, save_state_series_hdf5
from astrodyn_core.states.models import (
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
)
from astrodyn_core.states.orekit_dates import from_orekit_date, to_orekit_date
from astrodyn_core.states.orekit_resolvers import resolve_frame


def compile_scenario_maneuvers(
    scenario: ScenarioStateFile,
    initial_state: Any,
) -> tuple[CompiledManeuver, ...]:
    """Compile scenario maneuvers to absolute epochs + inertial delta-v vectors."""
    if not isinstance(scenario, ScenarioStateFile):
        raise TypeError("scenario must be a ScenarioStateFile.")
    if not hasattr(initial_state, "getDate") or not hasattr(initial_state, "getOrbit"):
        raise TypeError("initial_state must be an Orekit SpacecraftState-like object.")

    timeline = resolve_timeline_events(scenario.timeline, initial_state)
    compiled: list[CompiledManeuver] = []
    planning_state = initial_state

    for maneuver in scenario.maneuvers:
        trigger_type, event_date, trigger_metadata = resolve_maneuver_trigger(
            maneuver.trigger,
            planning_state,
            timeline,
        )
        event_state = keplerian_propagate_state(planning_state, event_date)
        dv_vec = resolve_delta_v_vector(maneuver.model, event_state, trigger_type)

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

        planning_state = apply_impulse_to_state(event_state, dv_vec)

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
            dv_vec = tuple_to_vector(maneuver.dv_inertial_mps)
            post_event_state = apply_impulse_to_state(pre_event_state, dv_vec)
            propagator.resetInitialState(post_event_state)
            maneuver_idx += 1

        state = propagator.propagate(sample_date)
        records_by_epoch[epoch] = state_to_record(
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


def keplerian_propagate_state(state: Any, target_date: Any):
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.analytical import KeplerianPropagator

    if float(target_date.durationFrom(state.getDate())) < 0.0:
        raise ValueError("target_date must be >= current state date for maneuver planning.")

    kp = KeplerianPropagator(state.getOrbit())
    propagated = kp.propagate(target_date)
    return SpacecraftState(propagated.getOrbit(), float(state.getMass()))


def apply_impulse_to_state(state: Any, delta_v: Any):
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


def state_to_record(
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
