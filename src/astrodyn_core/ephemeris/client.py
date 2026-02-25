"""High-level client API for ephemeris-based propagator workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from astrodyn_core.ephemeris.factory import create_propagator_from_spec as _create_propagator
from astrodyn_core.ephemeris.models import EphemerisFormat, EphemerisSpec
from astrodyn_core.ephemeris.parser import (
    parse_cpf as _parse_cpf,
    parse_ocm as _parse_ocm,
    parse_oem as _parse_oem,
    parse_sp3 as _parse_sp3,
)


@dataclass(slots=True)
class EphemerisClient:
    """Single entrypoint for ephemeris-based propagator creation.

    Supports creating Orekit ``BoundedPropagator`` objects from standard
    ephemeris file formats:

    - **OEM** (CCSDS Orbit Ephemeris Message) -- local files
    - **OCM** (CCSDS Orbit Comprehensive Message) -- local files
    - **SP3** (IGS Standard Product 3) -- downloaded via EDC FTP
    - **CPF** (ILRS Consolidated Prediction Format) -- downloaded via EDC API

    For remote formats (SP3, CPF), provide EDC credentials either via
    ``secrets_path`` (reads ``edc_username`` and ``edc_password`` from a
    ``secrets.ini`` file) or by passing pre-built client objects via
    ``edc_api_client`` and ``edc_ftp_client``.

    Args:
        secrets_path: Optional path to a ``secrets.ini`` file containing EDC
            credentials (``edc_username`` / ``edc_password``) for remote SP3/CPF
            workflows.
        cache_dir: Local cache directory for downloaded ephemeris files.
        edc_api_client: Optional pre-built EDC API client. When provided, this
            is used instead of lazily creating one from ``secrets_path``.
        edc_ftp_client: Optional pre-built EDC FTP client for SP3 downloads.

    Example:
        ```python
        client = EphemerisClient()
        propagator = client.create_propagator(
            EphemerisSpec.for_oem("data/oem_files/satellite.oem")
        )
        ```
    """

    secrets_path: str | Path | None = None
    cache_dir: str | Path = "data/cache"
    edc_api_client: Any | None = None
    edc_ftp_client: Any | None = None

    # ------------------------------------------------------------------
    # Primary workflow: spec -> propagator
    # ------------------------------------------------------------------

    def create_propagator(self, spec: EphemerisSpec) -> Any:
        """Create an Orekit bounded propagator from an ephemeris specification.

        Local specs (OEM/OCM) are parsed directly. Remote specs (SP3/CPF)
        trigger EDC metadata/file acquisition before parsing and propagator
        construction.

        Args:
            spec: Immutable ephemeris request describing source, format, and
                required query or file fields.

        Returns:
            An Orekit ``BoundedPropagator`` (or equivalent Orekit propagator
            object returned by the parser/factory flow).

        Raises:
            ValueError: If required remote credentials/clients are missing or
                the spec cannot be fulfilled.
        """
        api_client = self._resolve_api_client(
            required=spec.source.value == "remote",
        )
        return _create_propagator(
            spec,
            edc_api_client=api_client,
            edc_ftp_client=self.edc_ftp_client,
            cache_dir=self.cache_dir,
        )

    def create_propagator_from_oem(self, file_path: str | Path) -> Any:
        """Create a propagator from a local OEM file.

        Args:
            file_path: Path to a local OEM file.

        Returns:
            An Orekit bounded propagator derived from the OEM ephemeris.
        """
        return self.create_propagator(EphemerisSpec.for_oem(file_path))

    def create_propagator_from_ocm(self, file_paths: str | Path | Sequence[str | Path]) -> Any:
        """Create a propagator from one or more local OCM files.

        Args:
            file_paths: One path or a sequence of OCM file paths.

        Returns:
            An Orekit bounded propagator, potentially an aggregate when multiple
            OCM segments/satellites are fused.
        """
        return self.create_propagator(EphemerisSpec.for_ocm(file_paths))

    def create_propagator_from_sp3(
        self,
        satellite_name: str,
        start_date: str,
        end_date: str,
        *,
        identifier_type: str = "satellite_name",
        provider_preference: Sequence[str] | None = None,
    ) -> Any:
        """Create a propagator from remote SP3 data via EDC FTP.

        Args:
            satellite_name: Satellite identifier value used for EDC lookup.
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.
            identifier_type: Identifier type field understood by EDC.
            provider_preference: Optional SP3 provider priority order.

        Returns:
            An Orekit bounded propagator built from one or more SP3 files.
        """
        spec = EphemerisSpec.for_sp3(
            satellite_name,
            start_date,
            end_date,
            identifier_type=identifier_type,
            provider_preference=provider_preference,
        )
        return self.create_propagator(spec)

    def create_propagator_from_cpf(
        self,
        satellite_name: str,
        start_date: str,
        end_date: str,
        *,
        identifier_type: str = "satellite_name",
        local_root: str | None = None,
    ) -> Any:
        """Create a propagator from remote CPF data via EDC API.

        Args:
            satellite_name: Satellite identifier value used for EDC lookup.
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.
            identifier_type: Identifier type field understood by EDC.
            local_root: Optional local CPF archive root checked before API
                download fallback.

        Returns:
            An Orekit bounded propagator built from one or more CPF files.
        """
        spec = EphemerisSpec.for_cpf(
            satellite_name,
            start_date,
            end_date,
            identifier_type=identifier_type,
            local_root=local_root,
        )
        return self.create_propagator(spec)

    # ------------------------------------------------------------------
    # Direct parsing (for advanced users)
    # ------------------------------------------------------------------

    def parse_oem(self, file_path: str | Path) -> Any:
        """Parse a local OEM file.

        Args:
            file_path: Path to a CCSDS OEM file.

        Returns:
            The parsed Orekit ``Oem`` object.
        """
        return _parse_oem(file_path)

    def parse_ocm(self, file_path: str | Path) -> Any:
        """Parse a local OCM file.

        Args:
            file_path: Path to a CCSDS OCM file.

        Returns:
            The parsed Orekit ``Ocm`` object.
        """
        return _parse_ocm(file_path)

    def parse_sp3(self, file_path: str | Path) -> Any:
        """Parse a local SP3 file.

        Args:
            file_path: Path to an SP3 file.

        Returns:
            The parsed Orekit ``SP3`` object.
        """
        return _parse_sp3(file_path)

    def parse_cpf(self, file_path: str | Path) -> Any:
        """Parse a local CPF file.

        Args:
            file_path: Path to an ILRS CPF file.

        Returns:
            The parsed Orekit ``CPF`` object.
        """
        return _parse_cpf(file_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_api_client(self, *, required: bool = True) -> Any | None:
        """Resolve or lazily construct an EDC API client.

        Args:
            required: If ``False``, return ``None`` when no API client is
                available and no credentials were supplied.

        Returns:
            The configured or newly created EDC API client, or ``None`` when not
            required and no client is available.

        Raises:
            ValueError: If a remote workflow requires credentials but neither
                ``secrets_path`` nor ``edc_api_client`` is available.
            FileNotFoundError: If ``secrets_path`` was provided but does not
                exist.
            KeyError: If the secrets file is missing required EDC credentials.
        """
        if self.edc_api_client is not None:
            return self.edc_api_client

        if not required:
            return None

        if self.secrets_path is None:
            raise ValueError(
                "EDC credentials required for remote ephemeris sources. "
                "Provide secrets_path or edc_api_client to EphemerisClient."
            )

        from astrodyn_core.ephemeris.downloader import EDCApiClient, load_edc_credentials

        username, password = load_edc_credentials(self.secrets_path)
        client = EDCApiClient(username, password)
        self.edc_api_client = client
        return client
