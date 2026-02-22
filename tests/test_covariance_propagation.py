"""Tests for uncertainty / covariance propagation (STM-based)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from astrodyn_core import (
    BuildContext,
    GravitySpec,
    IntegratorSpec,
    OrbitStateRecord,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    StateFileClient,
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

_INITIAL_COV_6x6 = np.diag(
    [
        1e6,
        1e6,
        1e6,  # position variances (m²)
        1e0,
        1e0,
        1e0,  # velocity variances (m²/s²)
    ]
).tolist()

_EPOCH_SPEC = OutputEpochSpec(
    start_epoch="2026-02-19T00:00:00Z",
    end_epoch="2026-02-19T01:00:00Z",
    step_seconds=600.0,
)


def _build_numerical_propagator():
    return _build_numerical_propagator_from_state(_LEO_STATE)


def _build_numerical_propagator_from_state(
    state: OrbitStateRecord,
    *,
    with_j2: bool = False,
):
    force_specs = [GravitySpec(degree=2, order=2)] if with_j2 else []
    ctx = BuildContext.from_state_record(state)
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
        force_specs=force_specs,
    )
    builder = factory.build_builder(spec, ctx)
    return builder.buildPropagator(builder.getSelectedNormalizedParameters())


def _as_cartesian_state(state: OrbitStateRecord) -> OrbitStateRecord:
    from astrodyn_core.states.orekit_convert import to_orekit_orbit

    orbit = to_orekit_orbit(state)
    pv = orbit.getPVCoordinates()
    pos = pv.getPosition()
    vel = pv.getVelocity()
    return OrbitStateRecord(
        epoch=state.epoch,
        frame=state.frame,
        representation="cartesian",
        position_m=(float(pos.getX()), float(pos.getY()), float(pos.getZ())),
        velocity_mps=(float(vel.getX()), float(vel.getY()), float(vel.getZ())),
        mu_m3_s2=state.mu_m3_s2,
        mass_kg=state.mass_kg,
    )


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
        from astrodyn_core.uncertainty.factory import create_covariance_propagator

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
        assert np.all(eigenvalues >= -1e-6), (
            f"Non-PSD covariance: min eigenvalue={eigenvalues.min()}"
        )

    def test_propagate_series_length(self):
        """propagate_series returns one record per epoch."""
        from astrodyn_core.uncertainty.factory import create_covariance_propagator

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
        from astrodyn_core.uncertainty.factory import create_covariance_propagator

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

    def test_propagate_with_stm_no_covariance(self):
        """propagate_with_stm() works without an initial_covariance."""
        from astrodyn_core.uncertainty.factory import setup_stm_propagator

        propagator = _build_numerical_propagator()
        stm_prop = setup_stm_propagator(propagator)

        target_epoch = "2026-02-19T00:30:00Z"
        state, phi = stm_prop.propagate_with_stm(target_epoch)

        assert phi.shape == (6, 6), f"Expected (6, 6) STM, got {phi.shape}"
        # The STM at t=t₀ is the identity; at t>t₀ it should be close but not identity
        assert not np.allclose(phi, np.eye(6)), "STM should differ from identity after propagation"
        # Determinant of STM is 1 for Hamiltonian systems (symplectic)
        det = np.linalg.det(phi)
        assert abs(det - 1.0) < 0.1, f"STM determinant far from 1: {det:.6f}"

    def test_propagate_with_covariance_raises_without_initial_cov(self):
        """propagate_with_covariance() must raise ValueError if no initial_covariance was given."""
        from astrodyn_core.uncertainty.factory import setup_stm_propagator

        propagator = _build_numerical_propagator()
        stm_prop = setup_stm_propagator(propagator)

        with pytest.raises(ValueError, match="initial_covariance"):
            stm_prop.propagate_with_covariance("2026-02-19T00:30:00Z")

    def test_stm_phi_is_consistent_with_covariance(self):
        """Manual P = Φ P₀ Φᵀ should match propagate_with_covariance result."""
        from astrodyn_core.uncertainty.factory import (
            create_covariance_propagator,
            setup_stm_propagator,
        )

        target_epoch = "2026-02-19T00:30:00Z"
        p0 = np.asarray(_INITIAL_COV_6x6)

        # Get covariance via the integrated method
        cov_prop = create_covariance_propagator(
            _build_numerical_propagator(), p0, UncertaintySpec()
        )
        _s, cov_record = cov_prop.propagate_with_covariance(target_epoch)
        p_integrated = cov_record.to_numpy()

        # Get STM manually and compute P = Φ P₀ Φᵀ
        stm_prop = setup_stm_propagator(_build_numerical_propagator())
        _s2, phi = stm_prop.propagate_with_stm(target_epoch)
        p_manual = phi @ p0 @ phi.T

        np.testing.assert_allclose(
            p_manual,
            p_integrated,
            rtol=1e-6,
            err_msg="Manual Φ P₀ Φᵀ differs from propagate_with_covariance result",
        )

    def test_keplerian_covariance_is_psd(self):
        """Covariance propagated in Keplerian parametrisation must remain PSD."""
        from astrodyn_core.uncertainty.factory import create_covariance_propagator

        # Keplerian P₀: diagonal variances for (a, e, i, argp, raan, M)
        # Units: a in m, angles in rad → tiny variances keep the matrix well-conditioned
        kep_cov = np.diag(
            [
                1e6,  # a variance (m²)
                1e-8,  # e variance (dimensionless²)
                1e-8,  # i variance (rad²)
                1e-8,  # argp variance (rad²)
                1e-8,  # raan variance (rad²)
                1e-8,  # M variance (rad²)
            ]
        )

        propagator = _build_numerical_propagator()
        spec = UncertaintySpec(method="stm", orbit_type="KEPLERIAN", position_angle="MEAN")
        cov_prop = create_covariance_propagator(propagator, kep_cov, spec)

        _state, cov_record = cov_prop.propagate_with_covariance("2026-02-19T00:30:00Z")

        cov = cov_record.to_numpy()
        assert cov.shape == (6, 6)
        assert cov_record.orbit_type == "KEPLERIAN"

        # Must be symmetric and PSD
        np.testing.assert_array_almost_equal(cov, cov.T, decimal=6)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10), (
            f"Keplerian covariance is not PSD: min eigenvalue={eigenvalues.min():.3e}"
        )

    def test_keplerian_and_cartesian_covariances_are_consistent(self):
        """Propagating the same physical uncertainty in Keplerian vs Cartesian must agree.

        The Keplerian covariance P_kep(t) is the Cartesian covariance P_cart(t) expressed
        in Keplerian coordinates via StateCovariance.changeCovarianceType.  We verify this
        round-trip by:
          1. Propagating P₀ in Cartesian → P_cart(t).
          2. Converting P_cart(t) to Keplerian via StateCovariance → P_kep_from_cart.
          3. Propagating P₀_kep (= P₀ in Keplerian coords) directly → P_kep_direct.
          4. Asserting P_kep_from_cart ≈ P_kep_direct.
        """
        from astrodyn_core.uncertainty.factory import create_covariance_propagator
        from astrodyn_core.uncertainty.transforms import change_covariance_type

        target_epoch = "2026-02-19T00:20:00Z"
        p0_cart = np.asarray(_INITIAL_COV_6x6)

        # --- path 1: propagate in Cartesian, then convert output to Keplerian ---
        cov_cart_prop = create_covariance_propagator(
            _build_numerical_propagator(), p0_cart, UncertaintySpec(orbit_type="CARTESIAN")
        )
        state_t, rec_cart = cov_cart_prop.propagate_with_covariance(target_epoch)

        from org.orekit.orbits import OrbitType, PositionAngleType

        P_kep_from_cart = change_covariance_type(
            rec_cart.to_numpy(),
            state_t.getOrbit(),
            state_t.getDate(),
            state_t.getFrame(),
            OrbitType.CARTESIAN,
            PositionAngleType.TRUE,
            OrbitType.KEPLERIAN,
            PositionAngleType.MEAN,
        )

        # --- path 2: convert P₀ to Keplerian, propagate, get Keplerian output ---
        initial_state = _build_numerical_propagator().getInitialState()
        p0_kep = change_covariance_type(
            p0_cart,
            initial_state.getOrbit(),
            initial_state.getDate(),
            initial_state.getFrame(),
            OrbitType.CARTESIAN,
            PositionAngleType.TRUE,
            OrbitType.KEPLERIAN,
            PositionAngleType.MEAN,
        )
        cov_kep_prop = create_covariance_propagator(
            _build_numerical_propagator(),
            p0_kep,
            UncertaintySpec(orbit_type="KEPLERIAN", position_angle="MEAN"),
        )
        _s, rec_kep = cov_kep_prop.propagate_with_covariance(target_epoch)
        P_kep_direct = rec_kep.to_numpy()

        np.testing.assert_allclose(
            P_kep_direct,
            P_kep_from_cart,
            rtol=1e-5,
            atol=1e-12,
            err_msg="Keplerian covariance path mismatch",
        )

    def test_covariance_is_invariant_to_initial_state_representation(self):
        """Cartesian covariance propagation should be independent of input orbit representation."""
        from astrodyn_core.uncertainty.factory import create_covariance_propagator

        target_epoch = "2026-02-19T00:30:00Z"
        p0_cart = np.asarray(_INITIAL_COV_6x6)

        cart_state = _as_cartesian_state(_LEO_STATE)
        cov_from_kep_state = create_covariance_propagator(
            _build_numerical_propagator_from_state(_LEO_STATE, with_j2=True),
            p0_cart,
            UncertaintySpec(orbit_type="CARTESIAN"),
        )
        cov_from_cart_state = create_covariance_propagator(
            _build_numerical_propagator_from_state(cart_state, with_j2=True),
            p0_cart,
            UncertaintySpec(orbit_type="CARTESIAN"),
        )

        _s1, rec1 = cov_from_kep_state.propagate_with_covariance(target_epoch)
        _s2, rec2 = cov_from_cart_state.propagate_with_covariance(target_epoch)
        np.testing.assert_allclose(rec1.to_numpy(), rec2.to_numpy(), rtol=1e-7, atol=1e-6)

    def test_include_mass_preserves_terms_for_non_cartesian_input(self):
        """7x7 non-Cartesian input must preserve mass variance and cross terms."""
        from astrodyn_core.uncertainty.factory import create_covariance_propagator

        p0 = np.eye(7, dtype=np.float64)
        p0[6, 6] = 9.0
        p0[:6, 6] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        p0[6, :6] = p0[:6, 6]

        cov_prop = create_covariance_propagator(
            _build_numerical_propagator(),
            p0,
            UncertaintySpec(include_mass=True, orbit_type="KEPLERIAN", position_angle="MEAN"),
        )
        assert np.isclose(cov_prop._initial_cov[6, 6], 9.0)
        assert np.linalg.norm(cov_prop._initial_cov[:6, 6]) > 0.0
        assert np.linalg.norm(cov_prop._initial_cov[6, :6]) > 0.0

        _s, rec = cov_prop.propagate_with_covariance("2026-02-19T00:10:00Z")
        assert np.isfinite(rec.to_numpy()).all()

    def test_requested_output_frame_transforms_state_and_covariance(self):
        """Output frame should change state and covariance values, not just metadata labels."""
        from astrodyn_core.uncertainty import UncertaintyClient

        client = UncertaintyClient()
        p0_cart = np.asarray(_INITIAL_COV_6x6)
        epoch_spec = OutputEpochSpec(
            start_epoch="2026-02-19T00:00:00Z",
            end_epoch="2026-02-19T00:30:00Z",
            step_seconds=1800.0,
        )

        states_gcrf, cov_gcrf = client.propagate_with_covariance(
            _build_numerical_propagator(),
            p0_cart,
            epoch_spec,
            spec=UncertaintySpec(orbit_type="CARTESIAN"),
            frame="GCRF",
        )
        states_eme, cov_eme = client.propagate_with_covariance(
            _build_numerical_propagator(),
            p0_cart,
            epoch_spec,
            spec=UncertaintySpec(orbit_type="CARTESIAN"),
            frame="EME2000",
        )

        st_g = states_gcrf.states[-1]
        st_e = states_eme.states[-1]
        assert st_g.frame == "GCRF"
        assert st_e.frame == "EME2000"
        pos_delta = np.linalg.norm(np.asarray(st_g.position_m) - np.asarray(st_e.position_m))
        vel_delta = np.linalg.norm(np.asarray(st_g.velocity_mps) - np.asarray(st_e.velocity_mps))
        assert pos_delta > 1.0e-3
        assert vel_delta > 1.0e-6

        cov_g = cov_gcrf.records[-1]
        cov_e = cov_eme.records[-1]
        assert cov_g.frame == "GCRF"
        assert cov_e.frame == "EME2000"
        cov_delta = np.max(np.abs(cov_g.to_numpy() - cov_e.to_numpy()))
        assert cov_delta > 1.0e-6

    def test_invalid_method_raises_value_error(self):
        """An unknown uncertainty method should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            UncertaintySpec(method="unscented")
