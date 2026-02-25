"""Ephemeris domain models for specification-driven propagator creation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class EphemerisFormat(str, Enum):
    """Supported ephemeris file formats.

    Attributes:
        OEM: CCSDS Orbit Ephemeris Message (local files).
        OCM: CCSDS Orbit Comprehensive Message (local file sets).
        SP3: IGS Standard Product 3 precise orbit files (remote via EDC FTP).
        CPF: ILRS Consolidated Prediction Format files (remote via EDC API).
    """

    OEM = "OEM"
    OCM = "OCM"
    SP3 = "SP3"
    CPF = "CPF"


class EphemerisSource(str, Enum):
    """Origin of ephemeris data for a request.

    Attributes:
        LOCAL: Data is read from one or more local files.
        REMOTE: Data is queried/downloaded via EDC clients before parsing.
    """

    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True, slots=True)
class EphemerisSpec:
    """Immutable specification for creating a BoundedPropagator from ephemeris data.

    This is the "request object" in the spec/factory pattern.  Create one using
    the convenience class methods (``for_oem``, ``for_ocm``, ``for_sp3``,
    ``for_cpf``), then pass it to ``EphemerisClient.create_propagator()``.

    Attributes:
        format: Ephemeris file format.
        source: Whether data is local or fetched remotely.
        file_path: Single local file path (OEM).
        file_paths: Multiple local file paths (OCM).
        satellite_name: Satellite identifier value for remote queries (SP3, CPF).
        identifier_type: Identifier type for remote queries (for example
            ``"satellite_name"`` or ``"cospar_id"``).
        start_date: Start date for remote queries in ``YYYY-MM-DD`` format.
        end_date: End date for remote queries in ``YYYY-MM-DD`` format.
        provider_preference: Preferred SP3 providers in priority order.
        local_root: Optional local CPF directory root used before API fallback.
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
        """Validate field combinations for the selected source/format.

        Raises:
            ValueError: If required fields are missing or the source/format
                combination is unsupported.
        """
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
        """Create a spec for a local OEM file.

        Args:
            file_path: Path to a CCSDS OEM (Orbit Ephemeris Message) file.

        Returns:
            A validated local OEM ephemeris specification.
        """
        return cls(
            format=EphemerisFormat.OEM,
            source=EphemerisSource.LOCAL,
            file_path=Path(file_path),
        )

    @classmethod
    def for_ocm(cls, file_paths: str | Path | Sequence[str | Path]) -> EphemerisSpec:
        """Create a spec for one or more local OCM files.

        Args:
            file_paths: One path or a sequence of paths to CCSDS OCM files.

        Returns:
            A validated local OCM ephemeris specification.
        """
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
        """Create a spec for remote SP3 data queried from EDC FTP.

        Args:
            satellite_name: Satellite identifier value (typically a satellite name).
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.
            identifier_type: Remote identifier field name understood by EDC.
            provider_preference: Optional provider priority order for SP3 file
                selection (one best file per day is chosen).

        Returns:
            A validated remote SP3 ephemeris specification.
        """
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
        """Create a spec for remote CPF data queried from the EDC API.

        Args:
            satellite_name: Satellite identifier value (for the chosen
                ``identifier_type``).
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.
            identifier_type: Remote identifier field name understood by EDC.
            local_root: Optional local CPF archive root to scan before API
                download fallback.

        Returns:
            A validated remote CPF ephemeris specification.
        """
        return cls(
            format=EphemerisFormat.CPF,
            source=EphemerisSource.REMOTE,
            satellite_name=satellite_name,
            identifier_type=identifier_type,
            start_date=start_date,
            end_date=end_date,
            local_root=local_root,
        )
