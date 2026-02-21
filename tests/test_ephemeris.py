"""Tests for the ephemeris module (models, spec validation, client wiring)."""

from __future__ import annotations

from pathlib import Path

import pytest

from astrodyn_core.ephemeris import EphemerisClient, EphemerisFormat, EphemerisSpec
from astrodyn_core.ephemeris.models import EphemerisSource


# ---------------------------------------------------------------------------
# EphemerisSpec model validation
# ---------------------------------------------------------------------------

class TestEphemerisSpec:
    """Spec validation and convenience constructors."""

    def test_for_oem_creates_local_spec(self, tmp_path: Path) -> None:
        path = tmp_path / "test.oem"
        path.write_text("dummy")
        spec = EphemerisSpec.for_oem(path)
        assert spec.format == EphemerisFormat.OEM
        assert spec.source == EphemerisSource.LOCAL
        assert spec.file_path == path

    def test_for_ocm_single_path(self, tmp_path: Path) -> None:
        path = tmp_path / "test.ocm"
        path.write_text("dummy")
        spec = EphemerisSpec.for_ocm(path)
        assert spec.format == EphemerisFormat.OCM
        assert spec.file_paths == (path,)

    def test_for_ocm_multiple_paths(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.ocm"
        p2 = tmp_path / "b.ocm"
        p1.write_text("a")
        p2.write_text("b")
        spec = EphemerisSpec.for_ocm([p1, p2])
        assert len(spec.file_paths) == 2

    def test_for_sp3_creates_remote_spec(self) -> None:
        spec = EphemerisSpec.for_sp3("lageos2", "2025-01-01", "2025-01-03")
        assert spec.format == EphemerisFormat.SP3
        assert spec.source == EphemerisSource.REMOTE
        assert spec.satellite_name == "lageos2"
        assert spec.start_date == "2025-01-01"
        assert spec.end_date == "2025-01-03"
        assert spec.identifier_type == "satellite_name"

    def test_for_sp3_with_provider_preference(self) -> None:
        spec = EphemerisSpec.for_sp3(
            "lageos2", "2025-01-01", "2025-01-03",
            provider_preference=["esa", "gfz"],
        )
        assert spec.provider_preference == ("esa", "gfz")

    def test_for_cpf_creates_remote_spec(self) -> None:
        spec = EphemerisSpec.for_cpf("lageos", "2025-01-01", "2025-01-03")
        assert spec.format == EphemerisFormat.CPF
        assert spec.source == EphemerisSource.REMOTE

    def test_for_cpf_with_local_root(self) -> None:
        spec = EphemerisSpec.for_cpf(
            "lageos", "2025-01-01", "2025-01-03",
            local_root="/data/cpf",
        )
        assert spec.local_root == "/data/cpf"

    def test_oem_requires_file_path(self) -> None:
        with pytest.raises(ValueError, match="file_path is required"):
            EphemerisSpec(
                format=EphemerisFormat.OEM,
                source=EphemerisSource.LOCAL,
            )

    def test_ocm_requires_file_paths(self) -> None:
        with pytest.raises(ValueError, match="file_paths is required"):
            EphemerisSpec(
                format=EphemerisFormat.OCM,
                source=EphemerisSource.LOCAL,
            )

    def test_local_rejects_sp3(self) -> None:
        with pytest.raises(ValueError, match="local source only supports"):
            EphemerisSpec(
                format=EphemerisFormat.SP3,
                source=EphemerisSource.LOCAL,
            )

    def test_remote_requires_identifier(self) -> None:
        with pytest.raises(ValueError, match="identifier_type and satellite_name"):
            EphemerisSpec(
                format=EphemerisFormat.SP3,
                source=EphemerisSource.REMOTE,
                start_date="2025-01-01",
                end_date="2025-01-03",
            )

    def test_remote_requires_dates(self) -> None:
        with pytest.raises(ValueError, match="start_date and end_date"):
            EphemerisSpec(
                format=EphemerisFormat.SP3,
                source=EphemerisSource.REMOTE,
                identifier_type="satellite_name",
                satellite_name="lageos2",
            )

    def test_remote_rejects_oem(self) -> None:
        with pytest.raises(ValueError, match="remote source only supports"):
            EphemerisSpec(
                format=EphemerisFormat.OEM,
                source=EphemerisSource.REMOTE,
                identifier_type="satellite_name",
                satellite_name="lageos",
                start_date="2025-01-01",
                end_date="2025-01-03",
            )

    def test_spec_is_frozen(self) -> None:
        spec = EphemerisSpec.for_sp3("lageos2", "2025-01-01", "2025-01-03")
        with pytest.raises(AttributeError):
            spec.format = EphemerisFormat.OEM  # type: ignore[misc]

    def test_ephemeris_format_values(self) -> None:
        assert EphemerisFormat.OEM.value == "OEM"
        assert EphemerisFormat.OCM.value == "OCM"
        assert EphemerisFormat.SP3.value == "SP3"
        assert EphemerisFormat.CPF.value == "CPF"


# ---------------------------------------------------------------------------
# EphemerisClient wiring
# ---------------------------------------------------------------------------

class TestEphemerisClient:
    """Client instantiation and method availability."""

    def test_default_client(self) -> None:
        client = EphemerisClient()
        assert client.secrets_path is None
        assert client.edc_api_client is None

    def test_client_has_create_methods(self) -> None:
        client = EphemerisClient()
        assert hasattr(client, "create_propagator")
        assert hasattr(client, "create_propagator_from_oem")
        assert hasattr(client, "create_propagator_from_ocm")
        assert hasattr(client, "create_propagator_from_sp3")
        assert hasattr(client, "create_propagator_from_cpf")

    def test_client_has_parse_methods(self) -> None:
        client = EphemerisClient()
        assert hasattr(client, "parse_oem")
        assert hasattr(client, "parse_ocm")
        assert hasattr(client, "parse_sp3")
        assert hasattr(client, "parse_cpf")

    def test_remote_requires_credentials(self) -> None:
        client = EphemerisClient()
        spec = EphemerisSpec.for_sp3("lageos2", "2025-01-01", "2025-01-03")
        with pytest.raises(ValueError, match="EDC credentials required"):
            client.create_propagator(spec)


# ---------------------------------------------------------------------------
# AstrodynClient integration
# ---------------------------------------------------------------------------

def test_astrodyn_client_has_ephemeris_facade() -> None:
    from astrodyn_core import AstrodynClient
    app = AstrodynClient()
    assert hasattr(app, "ephemeris")
    assert isinstance(app.ephemeris, EphemerisClient)


def test_astrodyn_client_passes_ephemeris_config() -> None:
    from astrodyn_core import AstrodynClient
    app = AstrodynClient(
        ephemeris_secrets_path="test_secrets.ini",
        ephemeris_cache_dir="/tmp/test_cache",
    )
    assert app.ephemeris.secrets_path == "test_secrets.ini"
    assert str(app.ephemeris.cache_dir) == "/tmp/test_cache"


# ---------------------------------------------------------------------------
# Root export consistency
# ---------------------------------------------------------------------------

def test_ephemeris_in_root_exports() -> None:
    import astrodyn_core
    assert "EphemerisClient" in astrodyn_core.__all__
    assert "EphemerisSpec" in astrodyn_core.__all__
    assert "EphemerisFormat" in astrodyn_core.__all__
    assert hasattr(astrodyn_core, "EphemerisClient")
    assert hasattr(astrodyn_core, "EphemerisSpec")
    assert hasattr(astrodyn_core, "EphemerisFormat")
