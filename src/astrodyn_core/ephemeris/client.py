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

    Examples
    --------
    Local OEM file::

        client = EphemerisClient()
        propagator = client.create_propagator(
            EphemerisSpec.for_oem("data/oem_files/satellite.oem")
        )

    Remote SP3 via EDC::

        client = EphemerisClient(secrets_path="secrets.ini")
        spec = EphemerisSpec.for_sp3("lageos2", "2025-01-01", "2025-01-03")
        propagator = client.create_propagator(spec)
    """

    secrets_path: str | Path | None = None
    cache_dir: str | Path = "data/cache"
    edc_api_client: Any | None = None
    edc_ftp_client: Any | None = None

    # ------------------------------------------------------------------
    # Primary workflow: spec -> propagator
    # ------------------------------------------------------------------

    def create_propagator(self, spec: EphemerisSpec) -> Any:
        """Create a BoundedPropagator from an EphemerisSpec.

        For local specs (OEM, OCM), returns immediately.
        For remote specs (SP3, CPF), acquires data from EDC first.

        Returns an Orekit ``BoundedPropagator``.
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
        """Convenience: create a propagator from a local OEM file."""
        return self.create_propagator(EphemerisSpec.for_oem(file_path))

    def create_propagator_from_ocm(self, file_paths: str | Path | Sequence[str | Path]) -> Any:
        """Convenience: create a propagator from local OCM file(s)."""
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
        """Convenience: create a propagator from SP3 data via EDC FTP."""
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
        """Convenience: create a propagator from CPF data via EDC API."""
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
        """Parse an OEM file and return the Orekit ``Oem`` object."""
        return _parse_oem(file_path)

    def parse_ocm(self, file_path: str | Path) -> Any:
        """Parse an OCM file and return the Orekit ``Ocm`` object."""
        return _parse_ocm(file_path)

    def parse_sp3(self, file_path: str | Path) -> Any:
        """Parse an SP3 file and return the Orekit ``SP3`` object."""
        return _parse_sp3(file_path)

    def parse_cpf(self, file_path: str | Path) -> Any:
        """Parse a CPF file and return the Orekit ``CPF`` object."""
        return _parse_cpf(file_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_api_client(self, *, required: bool = True) -> Any | None:
        """Lazily resolve or create the EDC API client."""
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
