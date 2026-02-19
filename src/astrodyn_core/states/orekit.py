"""Conversion helpers between state-file records and Orekit objects."""

from __future__ import annotations

import math
from datetime import timezone
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.propagation.config import get_itrf_frame, get_mu
from astrodyn_core.states.io import save_state_series_compact_with_style, save_state_series_hdf5
from astrodyn_core.states.models import (
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
)
from astrodyn_core.states.validation import parse_epoch_utc



def to_orekit_orbit(
    record: OrbitStateRecord,
    universe: Mapping[str, Any] | None = None,
):
    """Convert an OrbitStateRecord into an Orekit Orbit instance."""
    if not isinstance(record, OrbitStateRecord):
        raise TypeError("record must be an OrbitStateRecord.")

    try:
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit, PositionAngleType
        from org.orekit.utils import PVCoordinates
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    frame = resolve_frame(record.frame, universe=universe)
    date = to_orekit_date(record.epoch)
    mu = resolve_mu(record.mu_m3_s2, universe=universe)

    if record.representation == "cartesian":
        pos = Vector3D(*record.position_m)
        vel = Vector3D(*record.velocity_mps)
        pv = PVCoordinates(pos, vel)
        return CartesianOrbit(pv, frame, date, mu)

    if record.representation == "keplerian":
        el = record.elements or {}
        angle_type = getattr(PositionAngleType, el["anomaly_type"])
        return KeplerianOrbit(
            el["a_m"],
            el["e"],
            math.radians(el["i_deg"]),
            math.radians(el["argp_deg"]),
            math.radians(el["raan_deg"]),
            math.radians(el["anomaly_deg"]),
            angle_type,
            frame,
            date,
            mu,
        )

    if record.representation == "equinoctial":
        el = record.elements or {}
        angle_type = getattr(PositionAngleType, el["anomaly_type"])
        return EquinoctialOrbit(
            el["a_m"],
            el["ex"],
            el["ey"],
            el["hx"],
            el["hy"],
            math.radians(el["l_deg"]),
            angle_type,
            frame,
            date,
            mu,
        )

    raise ValueError(f"Unsupported representation '{record.representation}'.")


def to_orekit_date(epoch: str):
    """Convert an ISO-8601 epoch string into Orekit AbsoluteDate (UTC)."""
    parsed = parse_epoch_utc(epoch)
    try:
        from orekit.pyhelpers import datetime_to_absolutedate
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    return datetime_to_absolutedate(parsed)


def from_orekit_date(date: Any) -> str:
    """Convert Orekit AbsoluteDate into an ISO-8601 UTC epoch string."""
    try:
        from orekit.pyhelpers import absolutedate_to_datetime
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    parsed = absolutedate_to_datetime(date, tz_aware=True)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_frame(frame_name: str, universe: Mapping[str, Any] | None = None):
    """Resolve a frame name from state files into an Orekit frame."""
    try:
        from org.orekit.frames import FramesFactory
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    normalized = frame_name.strip().upper()
    if normalized == "GCRF":
        return FramesFactory.getGCRF()
    if normalized == "EME2000":
        return FramesFactory.getEME2000()
    if normalized == "TEME":
        return FramesFactory.getTEME()

    if normalized in {"ITRF", "ITRF_2020", "ITRF_2014", "ITRF_2008"}:
        cfg = dict(universe or {})
        if normalized != "ITRF":
            cfg["itrf_version"] = normalized
        return get_itrf_frame(cfg if cfg else None)

    raise ValueError(f"Unsupported frame '{frame_name}'.")


def resolve_mu(mu_m3_s2: float | str, universe: Mapping[str, Any] | None = None) -> float:
    """Resolve a numeric gravitational parameter in m^3/s^2."""
    if isinstance(mu_m3_s2, (int, float)):
        return float(mu_m3_s2)

    if isinstance(mu_m3_s2, str):
        cfg = dict(universe or {})
        cfg["gravitational_parameter"] = mu_m3_s2.strip().upper()
        return get_mu(cfg)

    raise TypeError("mu_m3_s2 must be a float or predefined model string.")


def state_series_to_ephemeris(
    series: StateSeries,
    *,
    universe: Mapping[str, Any] | None = None,
    interpolation_samples: int | None = None,
    default_mass_kg: float = 1000.0,
):
    """Convert a StateSeries into an Orekit Ephemeris (bounded propagator)."""
    if not isinstance(series, StateSeries):
        raise TypeError("series must be a StateSeries.")

    try:
        from java.util import ArrayList
        from org.orekit.propagation import SpacecraftState
        from org.orekit.propagation.analytical import Ephemeris
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    sorted_records = sorted(series.states, key=lambda item: parse_epoch_utc(item.epoch))

    states = ArrayList()
    for record in sorted_records:
        orbit = to_orekit_orbit(record, universe=universe)
        mass = float(record.mass_kg) if record.mass_kg is not None else float(default_mass_kg)
        states.add(SpacecraftState(orbit, mass))

    samples = _resolve_interpolation_samples(series, interpolation_samples)
    return Ephemeris(states, samples)


def scenario_to_ephemeris(
    scenario: ScenarioStateFile,
    *,
    series_name: str | None = None,
    interpolation_samples: int | None = None,
    default_mass_kg: float = 1000.0,
):
    """Convert one state series from a ScenarioStateFile into an Orekit Ephemeris."""
    if not isinstance(scenario, ScenarioStateFile):
        raise TypeError("scenario must be a ScenarioStateFile.")
    if not scenario.state_series:
        raise ValueError("Scenario has no state_series to convert.")

    selected: StateSeries | None = None
    if series_name is None:
        selected = scenario.state_series[0]
    else:
        for candidate in scenario.state_series:
            if candidate.name == series_name:
                selected = candidate
                break
        if selected is None:
            raise ValueError(f"State series '{series_name}' was not found.")

    return state_series_to_ephemeris(
        selected,
        universe=scenario.universe,
        interpolation_samples=interpolation_samples,
        default_mass_kg=default_mass_kg,
    )


def _resolve_interpolation_samples(series: StateSeries, explicit: int | None) -> int:
    if explicit is not None:
        if explicit < 2:
            raise ValueError("interpolation_samples must be >= 2.")
        return int(explicit)

    interpolation_cfg = series.interpolation
    samples = interpolation_cfg.get("samples")
    if samples is not None:
        samples_int = int(samples)
        if samples_int < 2:
            raise ValueError("state_series.interpolation.samples must be >= 2.")
        return samples_int

    hint = (series.interpolation_hint or "").strip().lower()
    hint_to_samples = {
        "linear": 2,
        "coarse": 4,
        "medium": 6,
        "lagrange": 8,
        "fine": 10,
    }
    if hint in hint_to_samples:
        return hint_to_samples[hint]

    return 8


def export_trajectory_from_propagator(
    propagator: Any,
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
) -> Path:
    """Export sampled states to YAML/HDF5 from a propagator or precomputed ephemeris."""
    if not isinstance(epoch_spec, OutputEpochSpec):
        raise TypeError("epoch_spec must be an OutputEpochSpec.")

    epochs = epoch_spec.epochs()
    if not epochs:
        raise ValueError("epoch_spec produced no epochs.")

    ephemeris = _resolve_sampling_ephemeris(propagator, epochs)
    sample_dates = [to_orekit_date(epoch) for epoch in epochs]
    _validate_requested_epochs(ephemeris, sample_dates, epochs)

    output_frame = resolve_frame(frame, universe=universe)
    rep = representation.strip().lower()
    if rep not in {"cartesian", "keplerian", "equinoctial"}:
        raise ValueError("representation must be one of {'cartesian', 'keplerian', 'equinoctial'}.")

    source_is_ephemeris = _is_precomputed_ephemeris(ephemeris)
    records: list[OrbitStateRecord] = []
    for epoch, date in zip(epochs, sample_dates):
        try:
            state = ephemeris.propagate(date)
        except Exception as exc:
            if source_is_ephemeris:
                raise ValueError(
                    f"Requested epoch '{epoch}' is outside the available range of the provided ephemeris."
                ) from exc
            raise
        records.append(
            _state_to_record(
                state,
                epoch=epoch,
                representation=rep,
                frame_name=frame,
                output_frame=output_frame,
                mu_m3_s2=mu_m3_s2,
                default_mass_kg=default_mass_kg,
            )
        )

    series = StateSeries(
        name=series_name,
        states=tuple(records),
        interpolation={"method": "orekit_ephemeris", "samples": int(interpolation_samples)},
    )

    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return save_state_series_hdf5(path, series)
    return save_state_series_compact_with_style(path, series, dense_rows=dense_yaml)


def _resolve_sampling_ephemeris(source: Any, epochs: tuple[str, ...]) -> Any:
    if _is_precomputed_ephemeris(source):
        return source

    if hasattr(source, "getEphemerisGenerator"):
        end_epoch = max(epochs, key=parse_epoch_utc)
        generator = source.getEphemerisGenerator()
        source.propagate(to_orekit_date(end_epoch))
        return generator.getGeneratedEphemeris()

    if hasattr(source, "propagate"):
        return source

    raise TypeError(
        "source must be an Orekit propagator with getEphemerisGenerator() "
        "or a precomputed ephemeris/bounded propagator exposing propagate()."
    )


def _is_precomputed_ephemeris(source: Any) -> bool:
    source_type = type(source)
    type_name = str(getattr(source_type, "__name__", "")).lower()
    type_module = str(getattr(source_type, "__module__", "")).lower()
    full_name = f"{type_module}.{type_name}"
    return (
        "boundedpropagator" in full_name
        or type_name == "ephemeris"
        or full_name.endswith(".ephemeris")
    )


def _validate_requested_epochs(ephemeris: Any, dates: list[Any], epochs: tuple[str, ...]) -> None:
    if not (hasattr(ephemeris, "getMinDate") and hasattr(ephemeris, "getMaxDate")):
        return

    min_date = ephemeris.getMinDate()
    max_date = ephemeris.getMaxDate()
    min_epoch = from_orekit_date(min_date)
    max_epoch = from_orekit_date(max_date)

    for epoch, date in zip(epochs, dates):
        if float(date.durationFrom(min_date)) < 0.0 or float(date.durationFrom(max_date)) > 0.0:
            raise ValueError(
                f"Requested epoch '{epoch}' is outside ephemeris bounds "
                f"[{min_epoch}, {max_epoch}]."
            )


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
    try:
        from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

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
