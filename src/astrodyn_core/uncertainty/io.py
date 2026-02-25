"""Covariance series I/O: YAML and HDF5 formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries


# ---------------------------------------------------------------------------
# YAML / JSON
# ---------------------------------------------------------------------------

def save_covariance_series_yaml(
    path: str | Path,
    series: CovarianceSeries,
) -> Path:
    """Save a :class:`CovarianceSeries` to a YAML file.

    The output format is::

        name: <series.name>
        method: stm
        records:
          - epoch: "..."
            frame: GCRF
            orbit_type: CARTESIAN
            include_mass: false
            matrix:
              - [<row0>]
              - [<row1>]
              ...

    Args:
        path: Destination file path (``.yaml`` or ``.json``).
        series: The covariance series to save.

    Returns:
        The resolved output path.

    Raises:
        ImportError: If ``PyYAML`` is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = series.to_mapping()

    with out.open("w", encoding="utf-8") as fh:
        yaml.dump(payload, fh, default_flow_style=None, sort_keys=False, allow_unicode=True)

    return out


def load_covariance_series_yaml(path: str | Path) -> CovarianceSeries:
    """Load a :class:`CovarianceSeries` from a YAML file.

    Args:
        path: Source file path (``.yaml`` or ``.json``).

    Returns:
        Loaded covariance series.

    Raises:
        ImportError: If ``PyYAML`` is not installed.
        FileNotFoundError: If the source file does not exist.
        ValueError: If the root YAML object is not a mapping.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Covariance file not found: {src}")

    with src.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at root of {src}, got {type(data).__name__}.")

    return CovarianceSeries.from_mapping(data)


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

_HDF5_EPOCHS_DS = "epochs"
_HDF5_MATRICES_DS = "matrices"


def save_covariance_series_hdf5(
    path: str | Path,
    series: CovarianceSeries,
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Save a :class:`CovarianceSeries` to an HDF5 file.

    Layout::

        /epochs          - string dataset, shape (N,)
        /matrices        - float64 dataset, shape (N, n, n)
        attributes:
            name, method, frame, orbit_type, include_mass

    Args:
        path: Destination ``.h5`` or ``.hdf5`` file path.
        series: The covariance series to save.
        compression: HDF5 compression filter (``"gzip"`` default, ``None`` to
            disable).
        compression_opts: Compression level (1-9 for gzip).

    Returns:
        The resolved output path.

    Raises:
        ImportError: If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required: pip install h5py") from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([r.epoch for r in series.records], dtype=object)
    matrices = series.matrices_numpy()  # shape (N, n, n)

    # Check consistency
    first = series.records[0]
    frame = first.frame
    orbit_type = first.orbit_type
    include_mass = first.include_mass

    compress_kwargs: dict[str, Any] = {}
    if compression is not None:
        compress_kwargs["compression"] = compression
        compress_kwargs["compression_opts"] = compression_opts

    with h5py.File(out, "w") as fh:
        fh.create_dataset(_HDF5_EPOCHS_DS, data=epochs, dtype=h5py.special_dtype(vlen=str))
        fh.create_dataset(_HDF5_MATRICES_DS, data=matrices, dtype=np.float64, **compress_kwargs)

        fh.attrs["name"] = series.name
        fh.attrs["method"] = series.method
        fh.attrs["frame"] = frame
        fh.attrs["orbit_type"] = orbit_type
        fh.attrs["include_mass"] = int(include_mass)

    return out


def load_covariance_series_hdf5(path: str | Path) -> CovarianceSeries:
    """Load a :class:`CovarianceSeries` from an HDF5 file.

    Args:
        path: Source ``.h5`` or ``.hdf5`` file path.

    Returns:
        Loaded covariance series.

    Raises:
        ImportError: If ``h5py`` is not installed.
        FileNotFoundError: If the source file does not exist.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required: pip install h5py") from exc

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Covariance HDF5 file not found: {src}")

    with h5py.File(src, "r") as fh:
        name = str(fh.attrs.get("name", "covariance"))
        method = str(fh.attrs.get("method", "stm"))
        frame = str(fh.attrs.get("frame", "GCRF"))
        orbit_type = str(fh.attrs.get("orbit_type", "CARTESIAN"))
        include_mass = bool(int(fh.attrs.get("include_mass", 0)))

        epochs_raw = fh[_HDF5_EPOCHS_DS][()]
        matrices_raw = fh[_HDF5_MATRICES_DS][()]

    epochs = [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in epochs_raw]
    records = tuple(
        CovarianceRecord.from_numpy(
            epoch=epoch,
            matrix=matrices_raw[i],
            frame=frame,
            orbit_type=orbit_type,
            include_mass=include_mass,
        )
        for i, epoch in enumerate(epochs)
    )

    return CovarianceSeries(name=name, records=records, method=method)


# ---------------------------------------------------------------------------
# Auto-dispatch by extension
# ---------------------------------------------------------------------------

def save_covariance_series(
    path: str | Path,
    series: CovarianceSeries,
    **kwargs: Any,
) -> Path:
    """Save covariance series, auto-selecting format from file extension.

    Uses HDF5 for ``.h5``/``.hdf5`` and YAML for everything else.

    Args:
        path: Destination file path.
        series: Covariance series to save.
        **kwargs: Format-specific keyword arguments forwarded to the selected
            backend.

    Returns:
        Resolved output path.
    """
    p = Path(path)
    if p.suffix.lower() in {".h5", ".hdf5"}:
        return save_covariance_series_hdf5(p, series, **kwargs)
    return save_covariance_series_yaml(p, series)


def load_covariance_series(path: str | Path) -> CovarianceSeries:
    """Load a covariance series by auto-detecting the file format.

    Args:
        path: Source file path.

    Returns:
        Loaded covariance series.
    """
    p = Path(path)
    if p.suffix.lower() in {".h5", ".hdf5"}:
        return load_covariance_series_hdf5(p)
    return load_covariance_series_yaml(p)
