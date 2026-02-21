"""Ephemeris domain models for specification-driven propagator creation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class EphemerisFormat(str, Enum):
    """Supported ephemeris file formats."""

    OEM = "OEM"
    OCM = "OCM"
    SP3 = "SP3"
    CPF = "CPF"


class EphemerisSource(str, Enum):
    """Where ephemeris data comes from."""

    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True, slots=True)
class EphemerisSpec:
    """Immutable specification for creating a BoundedPropagator from ephemeris data.

    This is the "request object" in the spec/factory pattern.  Create one using
    the convenience class methods (``for_oem``, ``for_ocm``, ``for_sp3``,
    ``for_cpf``), then pass it to ``EphemerisClient.create_propagator()``.

    Parameters
    ----------
    format : EphemerisFormat
        Ephemeris file format.
    source : EphemerisSource
        Whether data is local or fetched remotely.
    file_path : Path, optional
        Single local file path (for OEM).
    file_paths : tuple of Path, optional
        Multiple local file paths (for OCM).
    satellite_name : str, optional
        Satellite name for remote queries (CPF, SP3).
    identifier_type : str, optional
        Type of satellite identifier for remote queries
        (e.g. "satellite_name", "cospar_id").
    start_date : str, optional
        Start date in YYYY-MM-DD format (remote queries).
    end_date : str, optional
        End date in YYYY-MM-DD format (remote queries).
    provider_preference : tuple of str, optional
        Preferred SP3 data providers in order of preference.
    local_root : str, optional
        Root directory for local CPF file cache.
    """

    format: EphemerisFormat
    source: EphemerisSource
    file_path: Path | None = None
    file_paths: tuple[Path, ...] | None = None
    satellite_name: str | None = None
    identifier_type: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    provider_preference: tuple[str, ...] | None = None
    local_root: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate field combinations."""
        if self.source == EphemerisSource.LOCAL:
            if self.format == EphemerisFormat.OEM:
                if self.file_path is None:
                    raise ValueError("EphemerisSpec: file_path is required for OEM.")
            elif self.format == EphemerisFormat.OCM:
                if not self.file_paths:
                    raise ValueError("EphemerisSpec: file_paths is required for OCM.")
            else:
                raise ValueError(
                    f"EphemerisSpec: local source only supports OEM or OCM, got {self.format.value}."
                )

        elif self.source == EphemerisSource.REMOTE:
            if self.format not in (EphemerisFormat.CPF, EphemerisFormat.SP3):
                raise ValueError(
                    f"EphemerisSpec: remote source only supports CPF or SP3, got {self.format.value}."
                )
            if not self.identifier_type or not self.satellite_name:
                raise ValueError(
                    "EphemerisSpec: identifier_type and satellite_name are required for remote sources."
                )
            if not self.start_date or not self.end_date:
                raise ValueError(
                    "EphemerisSpec: start_date and end_date are required for remote sources."
                )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_oem(cls, file_path: str | Path) -> EphemerisSpec:
        """Create a spec for a local OEM (CCSDS Orbit Ephemeris Message) file."""
        return cls(
            format=EphemerisFormat.OEM,
            source=EphemerisSource.LOCAL,
            file_path=Path(file_path),
        )

    @classmethod
    def for_ocm(cls, file_paths: str | Path | Sequence[str | Path]) -> EphemerisSpec:
        """Create a spec for local OCM (CCSDS Orbit Comprehensive Message) files."""
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        return cls(
            format=EphemerisFormat.OCM,
            source=EphemerisSource.LOCAL,
            file_paths=tuple(Path(p) for p in file_paths),
        )

    @classmethod
    def for_sp3(
        cls,
        satellite_name: str,
        start_date: str,
        end_date: str,
        *,
        identifier_type: str = "satellite_name",
        provider_preference: Sequence[str] | None = None,
    ) -> EphemerisSpec:
        """Create a spec for remote SP3 (IGS Standard Product 3) data via EDC FTP."""
        return cls(
            format=EphemerisFormat.SP3,
            source=EphemerisSource.REMOTE,
            satellite_name=satellite_name,
            identifier_type=identifier_type,
            start_date=start_date,
            end_date=end_date,
            provider_preference=tuple(provider_preference) if provider_preference else None,
        )

    @classmethod
    def for_cpf(
        cls,
        satellite_name: str,
        start_date: str,
        end_date: str,
        *,
        identifier_type: str = "satellite_name",
        local_root: str | None = None,
    ) -> EphemerisSpec:
        """Create a spec for CPF (ILRS Consolidated Prediction Format) data via EDC API."""
        return cls(
            format=EphemerisFormat.CPF,
            source=EphemerisSource.REMOTE,
            satellite_name=satellite_name,
            identifier_type=identifier_type,
            start_date=start_date,
            end_date=end_date,
            local_root=local_root,
        )
