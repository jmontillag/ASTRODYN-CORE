"""Tests for the adaptive checkpoint-based GEqOE Taylor propagator."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from astrodyn_core.propagation.geqoe.core import (
    evaluate_cart_taylor,
    prepare_cart_coefficients,
)
from astrodyn_core.propagation.providers.geqoe.adaptive import (
    AdaptiveGEqOEPropagator,
    AdaptiveGEqOEProvider,
    Checkpoint,
)

_J2 = 0.0010826266835531513
_Re = 6378137.0
_mu = 3.986004418e14


def _reference_state() -> np.ndarray:
    return np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 1000.0], dtype=float)


def _body_constants() -> dict[str, float]:
    return {"j2": _J2, "re": _Re, "mu": _mu}


def _make_propagator(
    backend: str = "cpp",
    order: int = 4,
    pos_tol: float = 1.0,
    safety_factor: float = 0.8,
) -> AdaptiveGEqOEPropagator:
    return AdaptiveGEqOEPropagator(
        initial_orbit=None,
        body_constants=_body_constants(),
        order=order,
        backend=backend,
        pos_tol=pos_tol,
        safety_factor=safety_factor,
    )


def _init_propagator_with_state(
    backend: str = "cpp",
    order: int = 4,
    pos_tol: float = 1.0,
) -> AdaptiveGEqOEPropagator:
    """Create an adaptive propagator and manually initialise its state."""
    prop = _make_propagator(backend=backend, order=order, pos_tol=pos_tol)
    y0 = _reference_state()
    prop._y0 = y0
    prop._frame = None
    prop._epoch = None
    prop._mu = _mu
    prop._build_initial_checkpoint()
    return prop


# ---------------------------------------------------------------------------
# Checkpoint construction
# ---------------------------------------------------------------------------


class TestCheckpointConstruction:
    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_initial_checkpoint_has_identity_stm(self, backend):
        prop = _init_propagator_with_state(backend=backend)
        assert prop.num_checkpoints == 1
        ckpt = prop._checkpoints[0]
        npt.assert_allclose(ckpt.cumulative_stm, np.eye(6))
        assert ckpt.epoch_seconds == 0.0
        assert ckpt.dt_max > 0.0

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_dt_max_reasonable_for_order4(self, backend):
        prop = _init_propagator_with_state(backend=backend, order=4, pos_tol=1.0)
        ckpt = prop._checkpoints[0]
        # Order 4 with 1m tolerance should give at least tens of seconds
        assert ckpt.dt_max > 50.0

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_dt_max_increases_with_tolerance(self, backend):
        prop_tight = _init_propagator_with_state(backend=backend, pos_tol=0.1)
        prop_loose = _init_propagator_with_state(backend=backend, pos_tol=10.0)
        assert prop_loose._checkpoints[0].dt_max > prop_tight._checkpoints[0].dt_max


# ---------------------------------------------------------------------------
# Checkpoint extension
# ---------------------------------------------------------------------------


class TestCheckpointExtension:
    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_forward_extension(self, backend):
        prop = _init_propagator_with_state(backend=backend)
        dt_max = prop._checkpoints[0].dt_max
        # Query past the first checkpoint's range
        t_far = dt_max * 3.0
        ckpt = prop._find_or_extend(t_far)
        assert prop.num_checkpoints > 1
        assert abs(t_far - ckpt.epoch_seconds) <= ckpt.dt_max

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_backward_extension(self, backend):
        prop = _init_propagator_with_state(backend=backend)
        dt_max = prop._checkpoints[0].dt_max
        t_back = -dt_max * 2.0
        ckpt = prop._find_or_extend(t_back)
        assert prop.num_checkpoints > 1
        assert ckpt.epoch_seconds < 0.0

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_checkpoints_sorted(self, backend):
        prop = _init_propagator_with_state(backend=backend)
        dt_max = prop._checkpoints[0].dt_max
        # Extend both directions
        prop._find_or_extend(dt_max * 5)
        prop._find_or_extend(-dt_max * 3)
        epochs = [c.epoch_seconds for c in prop._checkpoints]
        assert epochs == sorted(epochs)

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_cached_checkpoint_reused(self, backend):
        prop = _init_propagator_with_state(backend=backend)
        ckpt1 = prop._find_or_extend(0.0)
        n1 = prop.num_checkpoints
        ckpt2 = prop._find_or_extend(0.0)
        n2 = prop.num_checkpoints
        assert n1 == n2  # no new checkpoint created
        assert ckpt1 is ckpt2


# ---------------------------------------------------------------------------
# Propagation accuracy (vs single-expansion baseline)
# ---------------------------------------------------------------------------


class TestPropagationAccuracy:
    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_short_arc_matches_single_expansion(self, backend):
        """Within a single checkpoint, adaptive should match single-expansion."""
        prop = _init_propagator_with_state(backend=backend, order=4)
        # Stay well within the initial checkpoint's dt_max
        dt_max = prop._checkpoints[0].dt_max
        dt = np.array([0.0, dt_max * 0.1, dt_max * 0.3, dt_max * 0.5])
        y_adap, stm_adap = prop.propagate_array(dt)

        # Single-expansion reference
        y0 = _reference_state()
        p = (_J2, _Re, _mu)
        coeffs, peq = prepare_cart_coefficients(y0, p, order=4)
        if backend == "cpp":
            from astrodyn_core.geqoe_cpp import (
                evaluate_cart_taylor_cpp,
                prepare_cart_coefficients_cpp,
            )

            coeffs_cpp, peq_cpp = prepare_cart_coefficients_cpp(
                y0, _J2, _Re, _mu, 4
            )
            y_ref, stm_ref = evaluate_cart_taylor_cpp(coeffs_cpp, peq_cpp, dt)
        else:
            y_ref, stm_ref = evaluate_cart_taylor(coeffs, peq, dt)

        npt.assert_allclose(y_adap, y_ref, rtol=1e-12, atol=1e-6)
        npt.assert_allclose(stm_adap, stm_ref, rtol=1e-10, atol=1e-6)

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_long_arc_position_bounded(self, backend):
        """Over 5 orbits, position error should stay bounded."""
        prop = _init_propagator_with_state(backend=backend, order=4, pos_tol=1.0)
        y0 = _reference_state()
        # ~5 orbits for a ~7000 km orbit (period ~ 5800s)
        r_mag = np.linalg.norm(y0[:3])
        period = 2.0 * np.pi * np.sqrt(r_mag**3 / _mu)
        dt = np.linspace(0, 5 * period, 200)
        y_adap, stm_adap = prop.propagate_array(dt)

        # Verify multiple checkpoints were used
        assert prop.num_checkpoints > 1

        # Verify states are physically reasonable (position magnitude within ~20%)
        for i in range(len(dt)):
            r = np.linalg.norm(y_adap[i, :3])
            assert 0.8 * r_mag < r < 1.2 * r_mag, (
                f"Position magnitude {r:.0f}m out of range at dt={dt[i]:.1f}s"
            )

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_stm_identity_at_epoch(self, backend):
        prop = _init_propagator_with_state(backend=backend)
        y, stm = prop.propagate_array(np.array([0.0]))
        npt.assert_allclose(stm[:, :, 0], np.eye(6), atol=1e-12)

    @pytest.mark.parametrize("backend", ["python", "cpp"])
    def test_stm_composition_invertible(self, backend):
        """STM determinant should stay near 1 (symplectic property)."""
        prop = _init_propagator_with_state(backend=backend, order=4, pos_tol=1.0)
        y0 = _reference_state()
        r_mag = np.linalg.norm(y0[:3])
        period = 2.0 * np.pi * np.sqrt(r_mag**3 / _mu)
        dt = np.array([0.0, period / 4, period / 2, period])
        _, stm = prop.propagate_array(dt)
        for i in range(len(dt)):
            det = np.linalg.det(stm[:, :, i])
            npt.assert_allclose(det, 1.0, atol=0.1,
                                err_msg=f"STM det at dt={dt[i]:.1f}s")


# ---------------------------------------------------------------------------
# Python vs C++ parity
# ---------------------------------------------------------------------------


class TestBackendParity:
    def test_propagate_array_parity(self):
        prop_py = _init_propagator_with_state(backend="python", order=4)
        prop_cpp = _init_propagator_with_state(backend="cpp", order=4)

        dt = np.array([0.0, 60.0, 300.0, 600.0])
        y_py, stm_py = prop_py.propagate_array(dt)
        y_cpp, stm_cpp = prop_cpp.propagate_array(dt)

        npt.assert_allclose(y_py, y_cpp, rtol=1e-12, atol=1e-6)
        npt.assert_allclose(stm_py, stm_cpp, rtol=1e-10, atol=1e-6)

    def test_long_arc_parity(self):
        """Python and C++ should produce very similar results over multiple checkpoints."""
        prop_py = _init_propagator_with_state(backend="python", order=4, pos_tol=1.0)
        prop_cpp = _init_propagator_with_state(backend="cpp", order=4, pos_tol=1.0)

        y0 = _reference_state()
        r_mag = np.linalg.norm(y0[:3])
        period = 2.0 * np.pi * np.sqrt(r_mag**3 / _mu)
        dt = np.linspace(0, 3 * period, 50)

        y_py, _ = prop_py.propagate_array(dt)
        y_cpp, _ = prop_cpp.propagate_array(dt)

        # Allow slightly looser tolerance for multi-checkpoint accumulated differences
        npt.assert_allclose(y_py, y_cpp, rtol=1e-10, atol=1e-3)


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


class TestAdaptiveGEqOEProvider:
    def test_provider_kind(self):
        p = AdaptiveGEqOEProvider()
        assert p.kind == "geqoe-adaptive"

    def test_provider_capabilities(self):
        p = AdaptiveGEqOEProvider()
        assert p.capabilities.supports_propagator is True
        assert p.capabilities.supports_stm is True
        assert p.capabilities.is_analytical is True
        assert p.capabilities.supports_builder is False

    def test_provider_registered(self):
        from astrodyn_core.propagation.registry import ProviderRegistry
        from astrodyn_core.propagation.providers import register_analytical_providers

        reg = ProviderRegistry()
        register_analytical_providers(reg)
        assert "geqoe-adaptive" in reg.available_propagator_kinds()
