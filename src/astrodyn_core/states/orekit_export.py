"""Trajectory export and ephemeris sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.states.io import save_state_series_compact_with_style, save_state_series_hdf5
from astrodyn_core.states.models import OutputEpochSpec, StateSeries
from astrodyn_core.states.orekit_convert import state_to_record
from astrodyn_core.states.orekit_dates import from_orekit_date, to_orekit_date
from astrodyn_core.states.orekit_resolvers import resolve_frame
from astrodyn_core.states.validation import parse_epoch_utc


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
    """Sample a propagator/ephemeris and export a serialized trajectory file.

    Args:
        propagator: Orekit propagator or ephemeris-like object exposing
            ``propagate`` (and optionally ``getEphemerisGenerator``).
        epoch_spec: Output epoch grid specification.
        output_path: Destination YAML/JSON/HDF5 path.
        series_name: Name for the exported state series.
        representation: Output representation (cartesian/keplerian/equinoctial).
        frame: Output frame name.
        mu_m3_s2: Gravitational parameter value/key stored in records.
        interpolation_samples: Interpolation metadata sample count for export.
        dense_yaml: Whether compact YAML rows should use flow-style rows.
        universe: Optional universe config used for frame resolution.
        default_mass_kg: Fallback mass when propagated states omit mass.

    Returns:
        Resolved output path.

    Raises:
        TypeError: If ``epoch_spec`` is invalid or source type is unsupported.
        ValueError: If requested epochs are empty/out of bounds or
            representation is unsupported.
    """
    if not isinstance(epoch_spec, OutputEpochSpec):
        raise TypeError("epoch_spec must be an OutputEpochSpec.")

    epochs = epoch_spec.epochs()
    if not epochs:
        raise ValueError("epoch_spec produced no epochs.")

    ephemeris = resolve_sampling_ephemeris(propagator, epochs)
    sample_dates = [to_orekit_date(epoch) for epoch in epochs]
    validate_requested_epochs(ephemeris, sample_dates, epochs)

    output_frame = resolve_frame(frame, universe=universe)
    rep = representation.strip().lower()
    if rep not in {"cartesian", "keplerian", "equinoctial"}:
        raise ValueError("representation must be one of {'cartesian', 'keplerian', 'equinoctial'}.")

    source_is_ephemeris = is_precomputed_ephemeris(ephemeris)
    records = []
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
            state_to_record(
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


def resolve_sampling_ephemeris(source: Any, epochs: tuple[str, ...]) -> Any:
    """Resolve a source object into something that can propagate requested epochs.

    Args:
        source: Orekit propagator, ephemeris, or bounded propagator-like object.
        epochs: Requested output epochs (used when generating an ephemeris).

    Returns:
        Object exposing ``propagate`` and optionally ephemeris bounds methods.

    Raises:
        TypeError: If ``source`` cannot be used for sampling.
    """
    if is_precomputed_ephemeris(source):
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


def is_precomputed_ephemeris(source: Any) -> bool:
    """Return whether ``source`` looks like a precomputed bounded ephemeris.

    Args:
        source: Candidate propagator-like object.

    Returns:
        ``True`` if the object appears to be an ephemeris/bounded propagator.
    """
    source_type = type(source)
    type_name = str(getattr(source_type, "__name__", "")).lower()
    type_module = str(getattr(source_type, "__module__", "")).lower()
    full_name = f"{type_module}.{type_name}"
    return (
        "boundedpropagator" in full_name
        or type_name == "ephemeris"
        or full_name.endswith(".ephemeris")
    )


def validate_requested_epochs(ephemeris: Any, dates: list[Any], epochs: tuple[str, ...]) -> None:
    """Validate that requested sample epochs lie within ephemeris bounds.

    Args:
        ephemeris: Ephemeris-like object, optionally exposing ``getMinDate`` and
            ``getMaxDate``.
        dates: Orekit dates corresponding to ``epochs``.
        epochs: Original epoch strings for error messages.

    Raises:
        ValueError: If any requested epoch is outside ephemeris bounds.
    """
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
