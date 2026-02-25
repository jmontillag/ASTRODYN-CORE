"""Load/save helpers for scenario state files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from astrodyn_core.states.models import OrbitStateRecord, ScenarioStateFile, StateSeries
from astrodyn_core.states.validation import parse_epoch_utc


def load_state_file(path: str | Path) -> ScenarioStateFile:
    """Load a YAML/JSON state file into a typed scenario model.

    Args:
        path: Source file path.

    Returns:
        Parsed ``ScenarioStateFile``.
    """
    data = _read_mapping(path)

    if _looks_like_orbit_state(data):
        return ScenarioStateFile(initial_state=OrbitStateRecord.from_mapping(data))
    return ScenarioStateFile.from_mapping(data)


def load_initial_state(path: str | Path) -> OrbitStateRecord:
    """Load only the initial state record from a state file.

    Args:
        path: Source file path.

    Returns:
        Initial orbit-state record.

    Raises:
        ValueError: If the file contains no ``initial_state``.
    """
    scenario = load_state_file(path)
    if scenario.initial_state is None:
        raise ValueError(f"No initial_state found in state file '{path}'.")
    return scenario.initial_state


def save_state_file(path: str | Path, scenario: ScenarioStateFile) -> Path:
    """Write a scenario state file as YAML or JSON based on extension.

    Args:
        path: Destination file path.
        scenario: Scenario model to serialize.

    Returns:
        Resolved output path.

    Raises:
        TypeError: If ``scenario`` is not a ``ScenarioStateFile``.
    """
    if not isinstance(scenario, ScenarioStateFile):
        raise TypeError("scenario must be a ScenarioStateFile instance.")
    payload = scenario.to_mapping()
    output_path = Path(path)
    _write_mapping(output_path, payload)
    return output_path


def save_initial_state(path: str | Path, state: OrbitStateRecord) -> Path:
    """Write a file containing only ``schema_version`` and ``initial_state``.

    Args:
        path: Destination file path.
        state: Orbit state record to serialize.

    Returns:
        Resolved output path.
    """
    scenario = ScenarioStateFile(initial_state=state)
    return save_state_file(path, scenario)


def save_state_series_compact(path: str | Path, series: StateSeries) -> Path:
    """Write one state series using the compact columns/rows schema.

    Args:
        path: Destination file path.
        series: State series to serialize.

    Returns:
        Resolved output path.
    """
    return save_state_series_compact_with_style(path, series, dense_rows=True)


def save_state_series_compact_with_style(
    path: str | Path,
    series: StateSeries,
    *,
    dense_rows: bool,
) -> Path:
    """Write one state series in compact format with configurable row style.

    Args:
        path: Destination file path.
        series: State series to serialize.
        dense_rows: Whether YAML row lists should use flow style.

    Returns:
        Resolved output path.

    Raises:
        TypeError: If ``series`` is not a ``StateSeries``.
    """
    if not isinstance(series, StateSeries):
        raise TypeError("series must be a StateSeries instance.")

    payload = {
        "schema_version": 1,
        "state_series": [state_series_to_compact_mapping(series)],
    }
    output_path = Path(path)
    _write_mapping(output_path, payload, flow_rows=dense_rows)
    return output_path


def state_series_to_compact_mapping(series: StateSeries) -> dict[str, Any]:
    """Convert a state series into the compact columns/rows mapping.

    Args:
        series: State series to convert.

    Returns:
        Compact mapping with ``defaults``/``columns``/``rows`` payload.

    Raises:
        ValueError: If the series is empty or has mixed representations/frames.
    """
    if not series.states:
        raise ValueError("StateSeries.states cannot be empty.")

    first = series.states[0]
    representation = first.representation
    frame = first.frame
    mu = first.mu_m3_s2

    for idx, record in enumerate(series.states[1:], start=1):
        if record.representation != representation:
            raise ValueError(
                f"StateSeries has mixed representations at index {idx}: "
                f"{representation} vs {record.representation}"
            )
        if record.frame != frame:
            raise ValueError(
                f"StateSeries has mixed frames at index {idx}: {frame} vs {record.frame}"
            )
        if record.mu_m3_s2 != mu:
            raise ValueError("StateSeries has mixed mu_m3_s2 values; compact export requires a single value.")

    columns = _series_columns(representation, include_mass=_has_any_mass(series.states))
    rows = [_record_to_row(record, columns) for record in series.states]

    mapping: dict[str, Any] = {
        "name": series.name,
        "interpolation_hint": series.interpolation_hint,
        "interpolation": dict(series.interpolation),
        "defaults": {
            "representation": representation,
            "frame": frame,
            "mu_m3_s2": mu,
        },
        "columns": columns,
        "rows": rows,
    }
    if mapping["interpolation_hint"] is None:
        mapping.pop("interpolation_hint")
    if not mapping["interpolation"]:
        mapping.pop("interpolation")

    return mapping


def save_state_series_hdf5(
    path: str | Path,
    series: StateSeries,
    *,
    compression: str = "gzip",
    compression_level: int = 4,
    shuffle: bool = True,
) -> Path:
    """Write one state series into an HDF5 file with columnar datasets.

    Args:
        path: Destination ``.h5``/``.hdf5`` path.
        series: State series to serialize.
        compression: HDF5 compression algorithm.
        compression_level: HDF5 compression level.
        shuffle: Whether to enable HDF5 shuffle filter.

    Returns:
        Resolved output path.

    Raises:
        TypeError: If ``series`` is not a ``StateSeries``.
        ValueError: If the series is empty.
        RuntimeError: If HDF5 dependencies are unavailable.
    """
    if not isinstance(series, StateSeries):
        raise TypeError("series must be a StateSeries instance.")

    mapping = state_series_to_compact_mapping(series)
    columns = list(mapping["columns"])
    rows = list(mapping["rows"])
    non_epoch_columns = columns[1:]

    if not rows:
        raise ValueError("StateSeries.states cannot be empty.")

    try:
        import h5py
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "h5py/numpy are required for HDF5 state-series I/O. Install them in the active environment."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_ns = np.asarray([_epoch_to_unix_ns(row[0]) for row in rows], dtype="int64")

    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = 1
        h5.attrs["series_name"] = str(mapping["name"])
        h5.attrs["defaults_json"] = json.dumps(mapping["defaults"])
        h5.attrs["interpolation_json"] = json.dumps(mapping.get("interpolation", {}))
        h5.attrs["interpolation_hint"] = str(mapping.get("interpolation_hint", ""))

        h5.create_dataset("epoch_unix_ns", data=epoch_ns)
        h5.create_dataset(
            "columns",
            data=np.asarray(non_epoch_columns, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        data_group = h5.create_group("data")
        for col_idx, column_name in enumerate(non_epoch_columns, start=1):
            values = [row[col_idx] for row in rows]
            if _all_numeric_or_none(values):
                array = np.asarray(
                    [float("nan") if value is None else float(value) for value in values], dtype="float64"
                )
                ds = data_group.create_dataset(
                    column_name,
                    data=array,
                    compression=compression,
                    compression_opts=compression_level,
                    shuffle=shuffle,
                )
                ds.attrs["kind"] = "numeric"
            else:
                text_values = ["" if value is None else str(value) for value in values]
                array = np.asarray(text_values, dtype=h5py.string_dtype(encoding="utf-8"))
                ds = data_group.create_dataset(column_name, data=array)
                ds.attrs["kind"] = "string"

    return output_path


def load_state_series_hdf5(path: str | Path) -> StateSeries:
    """Read one state series from an HDF5 file.

    Args:
        path: Source ``.h5``/``.hdf5`` file path.

    Returns:
        Loaded state series.

    Raises:
        RuntimeError: If HDF5 dependencies are unavailable.
        ValueError: If the file schema version or content is invalid.
    """
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "h5py is required for HDF5 state-series I/O. Install it in the active environment."
        ) from exc

    input_path = Path(path)
    with h5py.File(input_path, "r") as h5:
        if int(h5.attrs.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported schema_version in HDF5 file.")

        series_name = str(h5.attrs.get("series_name", "series"))
        defaults = json.loads(str(h5.attrs.get("defaults_json", "{}")))
        interpolation = json.loads(str(h5.attrs.get("interpolation_json", "{}")))
        interpolation_hint = str(h5.attrs.get("interpolation_hint", "")) or None

        epoch_ns = [int(value) for value in h5["epoch_unix_ns"][()]]
        non_epoch_columns = []
        for item in h5["columns"][()]:
            decoded = _decode_h5_str(item)
            if decoded is None:
                raise ValueError("HDF5 columns dataset contains an empty column name.")
            non_epoch_columns.append(decoded)
        data_group = h5["data"]

        rows: list[list[Any]] = []
        for row_idx, ns in enumerate(epoch_ns):
            row: list[Any] = [_unix_ns_to_epoch(ns)]
            for column_name in non_epoch_columns:
                dataset = data_group[column_name]
                kind = str(dataset.attrs.get("kind", "numeric"))
                value = dataset[row_idx]
                if kind == "numeric":
                    value_f = float(value)
                    row.append(None if _is_nan(value_f) else value_f)
                else:
                    row.append(_decode_h5_str(value))
            rows.append(row)

    return StateSeries.from_mapping(
        {
            "name": series_name,
            "interpolation_hint": interpolation_hint,
            "interpolation": interpolation,
            "defaults": defaults,
            "columns": ["epoch", *non_epoch_columns],
            "rows": rows,
        }
    )


def _read_mapping(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with open(file_path) as fh:
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(fh)
        elif file_path.suffix.lower() == ".json":
            raw = json.load(fh)
        else:
            raise ValueError("Unsupported file extension. Use .yaml, .yml, or .json")

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a mapping at top level, got {type(raw).__name__}.")
    return dict(raw)


def _write_mapping(path: Path, payload: dict[str, Any], *, flow_rows: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            if flow_rows:
                compact_payload = _coerce_rows_to_flow(payload)
                yaml.dump(compact_payload, fh, Dumper=_FlowRowsDumper, sort_keys=False)
            else:
                yaml.safe_dump(payload, fh, sort_keys=False)
        elif path.suffix.lower() == ".json":
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        else:
            raise ValueError("Unsupported file extension. Use .yaml, .yml, or .json")


def _looks_like_orbit_state(data: Mapping[str, Any]) -> bool:
    if "schema_version" in data:
        return False
    return "epoch" in data and "representation" in data


def _series_columns(representation: str, include_mass: bool) -> list[str]:
    if representation == "cartesian":
        columns = ["epoch", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]
    elif representation == "keplerian":
        columns = ["epoch", "a_m", "e", "i_deg", "argp_deg", "raan_deg", "anomaly_deg", "anomaly_type"]
    elif representation == "equinoctial":
        columns = ["epoch", "a_m", "ex", "ey", "hx", "hy", "l_deg", "anomaly_type"]
    else:
        raise ValueError(f"Unsupported representation '{representation}'.")

    if include_mass:
        columns.append("mass_kg")
    return columns


def _record_to_row(record: OrbitStateRecord, columns: list[str]) -> list[Any]:
    row: list[Any] = []
    elements = dict(record.elements or {})
    for column in columns:
        if column == "epoch":
            row.append(record.epoch)
            continue
        if column == "mass_kg":
            row.append(record.mass_kg)
            continue
        if column == "x_m":
            row.append(record.position_m[0])
            continue
        if column == "y_m":
            row.append(record.position_m[1])
            continue
        if column == "z_m":
            row.append(record.position_m[2])
            continue
        if column == "vx_mps":
            row.append(record.velocity_mps[0])
            continue
        if column == "vy_mps":
            row.append(record.velocity_mps[1])
            continue
        if column == "vz_mps":
            row.append(record.velocity_mps[2])
            continue
        row.append(elements.get(column))
    return row


def _has_any_mass(states: tuple[OrbitStateRecord, ...] | list[OrbitStateRecord]) -> bool:
    return any(record.mass_kg is not None for record in states)


class _FlowRow(list):
    """Marker list type to force YAML flow style for row arrays."""


class _FlowRowsDumper(yaml.SafeDumper):
    """YAML dumper with row-level flow style support."""


def _flow_row_representer(dumper: yaml.SafeDumper, data: _FlowRow):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_FlowRowsDumper.add_representer(_FlowRow, _flow_row_representer)


def _coerce_rows_to_flow(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    series_list = list(data.get("state_series", []))
    updated_series: list[dict[str, Any]] = []

    for item in series_list:
        if not isinstance(item, Mapping):
            updated_series.append(item)
            continue

        series_data = dict(item)
        rows = series_data.get("rows")
        if isinstance(rows, list):
            series_data["rows"] = [_FlowRow(row) if isinstance(row, list) else row for row in rows]
        updated_series.append(series_data)

    data["state_series"] = updated_series
    return data


def _epoch_to_unix_ns(epoch: str) -> int:
    dt = parse_epoch_utc(epoch)
    return int(dt.timestamp() * 1_000_000_000)


def _unix_ns_to_epoch(value: int) -> str:
    seconds = value / 1_000_000_000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _all_numeric_or_none(values: list[Any]) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            continue
        return False
    return True


def _decode_h5_str(value: Any) -> str | None:
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = str(value)
    if text == "":
        return None
    return text


def _is_nan(value: float) -> bool:
    return value != value
