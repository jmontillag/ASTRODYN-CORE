"""Covariance data models for uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from astrodyn_core.states.validation import parse_epoch_utc


@dataclass(frozen=True, slots=True)
class CovarianceRecord:
    """Covariance matrix snapshot at a single epoch.

    The ``matrix`` field stores an n×n covariance matrix (n=6 for orbit-only,
    n=7 if mass is included) as a tuple of tuples. Use ``to_numpy()`` for
    numerical operations.

    The covariance is expressed in the coordinate system defined by
    ``frame`` and ``orbit_type``.
    """

    epoch: str
    matrix: tuple[tuple[float, ...], ...]
    frame: str = "GCRF"
    orbit_type: str = "CARTESIAN"
    include_mass: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate epoch
        parse_epoch_utc(self.epoch)

        # Validate and normalize matrix shape
        mat = tuple(tuple(float(v) for v in row) for row in self.matrix)
        n = len(mat)
        expected_n = 7 if self.include_mass else 6
        if n != expected_n:
            raise ValueError(
                f"CovarianceRecord matrix must be {expected_n}×{expected_n} "
                f"(include_mass={self.include_mass}), got {n}×{len(mat[0]) if mat else 0}."
            )
        for i, row in enumerate(mat):
            if len(row) != n:
                raise ValueError(
                    f"CovarianceRecord matrix row {i} has length {len(row)}, expected {n}."
                )
        object.__setattr__(self, "matrix", mat)
        object.__setattr__(self, "frame", str(self.frame).strip().upper())
        object.__setattr__(self, "orbit_type", str(self.orbit_type).strip().upper())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_numpy(self) -> np.ndarray:
        """Return the covariance matrix as a numpy array of shape (n, n)."""
        return np.array(self.matrix, dtype=np.float64)

    @classmethod
    def from_numpy(
        cls,
        epoch: str,
        matrix: np.ndarray,
        *,
        frame: str = "GCRF",
        orbit_type: str = "CARTESIAN",
        include_mass: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CovarianceRecord:
        """Build a CovarianceRecord from a numpy array."""
        mat_tuple = tuple(tuple(float(v) for v in row) for row in matrix)
        return cls(
            epoch=epoch,
            matrix=mat_tuple,
            frame=frame,
            orbit_type=orbit_type,
            include_mass=include_mass,
            metadata=metadata or {},
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CovarianceRecord:
        """Build from a plain dict (e.g., loaded from YAML)."""
        raw_mat = data.get("matrix", [])
        mat = tuple(tuple(float(v) for v in row) for row in raw_mat)
        return cls(
            epoch=str(data["epoch"]),
            matrix=mat,
            frame=str(data.get("frame", "GCRF")),
            orbit_type=str(data.get("orbit_type", "CARTESIAN")),
            include_mass=bool(data.get("include_mass", False)),
            metadata=data.get("metadata", {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epoch": self.epoch,
            "frame": self.frame,
            "orbit_type": self.orbit_type,
            "include_mass": self.include_mass,
            "matrix": [list(row) for row in self.matrix],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class CovarianceSeries:
    """Time series of covariance snapshots.

    Attributes
    ----------
    name:
        Identifier for this covariance series.
    records:
        Ordered tuple of :class:`CovarianceRecord` instances.
    method:
        Propagation method used to generate this series (``"stm"`` or
        ``"unscented"``). Informational only.
    """

    name: str
    records: tuple[CovarianceRecord, ...]
    method: str = "stm"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CovarianceSeries.name cannot be empty.")
        if not self.records:
            raise ValueError("CovarianceSeries.records cannot be empty.")
        object.__setattr__(self, "records", tuple(self.records))
        for rec in self.records:
            if not isinstance(rec, CovarianceRecord):
                raise TypeError("CovarianceSeries.records must contain CovarianceRecord values.")

    @property
    def epochs(self) -> tuple[str, ...]:
        """Return all epochs in the series."""
        return tuple(r.epoch for r in self.records)

    def matrices_numpy(self) -> np.ndarray:
        """Return all matrices stacked as shape (N, n, n) numpy array."""
        return np.stack([r.to_numpy() for r in self.records], axis=0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CovarianceSeries:
        records = tuple(CovarianceRecord.from_mapping(r) for r in data.get("records", []))
        return cls(
            name=str(data.get("name", "covariance")),
            records=records,
            method=str(data.get("method", "stm")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "records": [r.to_mapping() for r in self.records],
        }
