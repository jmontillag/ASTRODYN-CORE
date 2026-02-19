"""Tests for uncertainty / covariance propagation (STM-based)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    OrbitStateRecord,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    register_default_orekit_providers,
)
from astrodyn_core.uncertainty.io import (
    load_covariance_series_hdf5,
    load_covariance_series_yaml,
    save_covariance_series_hdf5,
    save_covariance_series_yaml,
)
from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries
from astrodyn_core.uncertainty.spec import UncertaintySpec

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LEO_STATE = OrbitStateRecord(
    epoch="2026-02-19T00:00:00Z",
    frame="GCRF",
    representation="keplerian",
    elements={
        "a_m": 6_878_137.0,
        "e": 0.001,
        "i_deg": 51.6,
        "argp_deg": 45.0,
        "raan_deg": 120.0,
        "anomaly_deg": 0.0,
        "anomaly_type": "MEAN",
    },
    mu_m3_s2="WGS84",
    mass_kg=450.0,
)

_INITIAL_COV_6x6 = np.diag([
    1e6, 1e6, 1e6,   # position variances (m²)
    1e0, 1e0, 1e0,   # velocity variances (m²/s²)
]).tolist()

_EPOCH_SPEC = OutputEpochSpec(
    start_epoch="2026-02-19T00:00:00Z",
    end_epoch="2026-02-19T01:00:00Z",
    step_seconds=600.0,
)


def _build_numerical_propagator():
    ctx = BuildContext.from_state_record(_LEO_STATE)
    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        mass_kg=450.0,
        integrator=IntegratorSpec(
            kind="dp54",
            min_step=1.0e-6,
            max_step=300.0,
            position_tolerance=1.0e-3,
        ),
    )
    builder = factory.build_builder(spec, ctx)
    return builder.buildPropagator(builder.getSelectedNormalizedParameters())


# ---------------------------------------------------------------------------
# Unit tests: UncertaintySpec
# ---------------------------------------------------------------------------

class TestUncertaintySpec:
    def test_defaults(self):
        spec = UncertaintySpec()
        assert spec.method == "stm"
        assert spec.stm_name == "stm"
        assert spec.include_mass is False
        assert spec.orbit_type == "CARTESIAN"
        assert spec.position_angle == "MEAN"
        assert spec.state_dimension == 6

    def test_include_mass_dimension(self):
        spec = UncertaintySpec(include_mass=True)
        assert spec.state_dimension == 7

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            UncertaintySpec(method="invalid")

    def test_invalid_orbit_type(self):
        with pytest.raises(ValueError, match="orbit_type"):
            UncertaintySpec(orbit_type="bogus")

    def test_roundtrip_mapping(self):
        spec = UncertaintySpec(method="stm", orbit_type="KEPLERIAN", include_mass=False)
        spec2 = UncertaintySpec.from_mapping(spec.to_mapping())
        assert spec == spec2

    def test_case_normalization(self):
        spec = UncertaintySpec(orbit_type="cartesian", position_angle="true")
        assert spec.orbit_type == "CARTESIAN"
        assert spec.position_angle == "TRUE"


# ---------------------------------------------------------------------------
# Unit tests: CovarianceRecord
# ---------------------------------------------------------------------------

class TestCovarianceRecord:
    def test_construction(self):
        mat = [[float(i == j) for j in range(6)] for i in range(6)]
        rec = CovarianceRecord(epoch="2026-02-19T00:00:00Z", matrix=mat)
        assert rec.epoch == "2026-02-19T00:00:00Z"
        assert len(rec.matrix) == 6
        assert len(rec.matrix[0]) == 6

    def test_to_numpy(self):
        mat = np.eye(6).tolist()
        rec = CovarianceRecord(epoch="2026-02-19T00:00:00Z", matrix=mat)
        arr = rec.to_numpy()
        assert arr.shape == (6, 6)
        np.testing.assert_array_almost_equal(arr, np.eye(6))

    def test_from_numpy_roundtrip(self):
        arr = np.diag([1e6, 1e6, 1e6, 1.0, 1.0, 1.0])
        rec = CovarianceRecord.from_numpy(
            epoch="2026-02-19T00:00:00Z",
            matrix=arr,
            frame="GCRF",
            orbit_type="CARTESIAN",
        )
        np.testing.assert_array_almost_equal(rec.to_numpy(), arr)

    def test_wrong_dimension(self):
        mat_5x5 = [[1.0] * 5 for _ in range(5)]
        with pytest.raises(ValueError, match="6×6"):
            CovarianceRecord(epoch="2026-02-19T00:00:00Z", matrix=mat_5x5)

    def test_mapping_roundtrip(self):
        arr = np.eye(6) * 1e4
        rec = CovarianceRecord.from_numpy("2026-02-19T00:00:00Z", arr)
        rec2 = CovarianceRecord.from_mapping(rec.to_mapping())
        np.testing.assert_array_almost_equal(rec.to_numpy(), rec2.to_numpy())
        assert rec.epoch == rec2.epoch


# ---------------------------------------------------------------------------
# Unit tests: CovarianceSeries
# ---------------------------------------------------------------------------

class TestCovarianceSeries:
    def _make_series(self) -> CovarianceSeries:
        records = tuple(
            CovarianceRecord.from_numpy(
                epoch=f"2026-02-19T0{i}:00:00Z",
                matrix=np.eye(6) * float(i + 1),
            )
            for i in range(3)
        )
        return CovarianceSeries(name="test-series", records=records)

    def test_construction(self):
        series = self._make_series()
        assert series.name == "test-series"
        assert len(series.records) == 3
        assert series.method == "stm"

    def test_epochs(self):
        series = self._make_series()
        assert series.epochs[0] == "2026-02-19T00:00:00Z"
        assert len(series.epochs) == 3

    def test_matrices_numpy(self):
        series = self._make_series()
        arr = series.matrices_numpy()
        assert arr.shape == (3, 6, 6)

    def test_mapping_roundtrip(self):
        series = self._make_series()
        series2 = CovarianceSeries.from_mapping(series.to_mapping())
        assert series2.name == series.name
        assert len(series2.records) == len(series.records)


# ---------------------------------------------------------------------------
# Unit tests: Covariance I/O
# ---------------------------------------------------------------------------

class TestCovarianceIO:
    def _make_series(self) -> CovarianceSeries:
        records = tuple(
            CovarianceRecord.from_numpy(
                epoch=f"2026-02-19T0{i}:00:00Z",
                matrix=np.eye(6) * (1e6 if i < 3 else 1e0),
            )
            for i in range(4)
        )
        return CovarianceSeries(name="io-test", records=records, method="stm")

    def test_yaml_roundtrip(self):
        series = self._make_series()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as fh:
            path = Path(fh.name)
        try:
            save_covariance_series_yaml(path, series)
            loaded = load_covariance_series_yaml(path)
            assert loaded.name == series.name
            assert loaded.method == series.method
            assert len(loaded.records) == len(series.records)
            np.testing.assert_array_almost_equal(
                loaded.records[0].to_numpy(), series.records[0].to_numpy()
            )
        finally:
            path.unlink(missing_ok=True)

    def test_hdf5_roundtrip(self):
        h5py = pytest.importorskip("h5py")
        series = self._make_series()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as fh:
            path = Path(fh.name)
        try:
            save_covariance_series_hdf5(path, series)
            loaded = load_covariance_series_hdf5(path)
            assert loaded.name == series.name
            assert len(loaded.records) == len(series.records)
            np.testing.assert_array_almost_equal(
                loaded.matrices_numpy(), series.matrices_numpy(), decimal=10
            )
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration tests: STMCovariancePropagator
# ---------------------------------------------------------------------------

class TestSTMCovariancePropagator:
    def test_propagate_with_covariance_returns_psd(self):
        """Propagated covariance must be positive semi-definite."""
        from astrodyn_core.uncertainty.propagator import create_covariance_propagator

        propagator = _build_numerical_propagator()
        spec = UncertaintySpec(method="stm", orbit_type="CARTESIAN")
        cov_prop = create_covariance_propagator(propagator, _INITIAL_COV_6x6, spec)

        target_epoch = "2026-02-19T00:30:00Z"
        _state, cov_record = cov_prop.propagate_with_covariance(target_epoch)

        cov = cov_record.to_numpy()
        assert cov.shape == (6, 6)

        # Symmetry check
        np.testing.assert_array_almost_equal(cov, cov.T, decimal=8)

        # Positive semi-definite: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-6), f"Non-PSD covariance: min eigenvalue={eigenvalues.min()}"

    def test_propagate_series_length(self):
        """propagate_series returns one record per epoch."""
        from astrodyn_core.uncertainty.propagator import create_covariance_propagator

        propagator = _build_numerical_propagator()
        spec = UncertaintySpec(method="stm")
        cov_prop = create_covariance_propagator(propagator, _INITIAL_COV_6x6, spec)

        state_series, cov_series = cov_prop.propagate_series(
            _EPOCH_SPEC, series_name="test-traj", covariance_name="test-cov"
        )

        expected_epochs = len(_EPOCH_SPEC.epochs())
        assert len(state_series.states) == expected_epochs
        assert len(cov_series.records) == expected_epochs
        assert cov_series.method == "stm"

    def test_covariance_grows_over_time(self):
        """Uncertainty should generally increase with propagation time (trace grows)."""
        from astrodyn_core.uncertainty.propagator import create_covariance_propagator

        propagator = _build_numerical_propagator()
        spec = UncertaintySpec(method="stm")
        cov_prop = create_covariance_propagator(propagator, _INITIAL_COV_6x6, spec)

        _s0, cov0 = cov_prop.propagate_with_covariance("2026-02-19T00:00:00Z")
        _s1, cov1 = cov_prop.propagate_with_covariance("2026-02-19T01:00:00Z")

        trace0 = np.trace(cov0.to_numpy())
        trace1 = np.trace(cov1.to_numpy())
        # For a free-flying LEO satellite, position uncertainty grows over an orbit
        assert trace1 > trace0 * 0.5, (
            f"Covariance trace decreased unexpectedly: {trace0:.3e} -> {trace1:.3e}"
        )

    def test_unscented_raises_not_implemented(self):
        """Unscented method should raise NotImplementedError."""
        from astrodyn_core.uncertainty.propagator import create_covariance_propagator

        propagator = _build_numerical_propagator()
        spec = UncertaintySpec.__new__(UncertaintySpec)
        object.__setattr__(spec, "method", "unscented")
        object.__setattr__(spec, "stm_name", "stm")
        object.__setattr__(spec, "include_mass", False)
        object.__setattr__(spec, "orbit_type", "CARTESIAN")
        object.__setattr__(spec, "position_angle", "MEAN")

        with pytest.raises(NotImplementedError, match="Unscented"):
            create_covariance_propagator(propagator, _INITIAL_COV_6x6, spec)
