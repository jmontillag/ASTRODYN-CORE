"""Tests for the GEqOE provider integration (Phase 5).

Tests are organized into two groups:

1. **Registry/factory integration** — no Orekit required.  Verifies that
   the provider registers, the factory resolves ``kind="geqoe"``, and
   capabilities are correctly reported.

2. **Orekit adapter** — skipped when Orekit is not installed.  Verifies
   ``propagate()`` returns valid ``SpacecraftState``, PV coordinates match
   the numpy engine, and ``get_native_state`` exposes the STM.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.factory import PropagatorFactory
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.providers.geqoe.provider import GEqOEProvider
from astrodyn_core.propagation.providers.geqoe.propagator import GEqOEPropagator
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.specs import PropagatorSpec


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_BODY_CONSTANTS = {
    "j2": 1.08262668e-3,
    "re": 6378137.0,
    "mu": 3.986004418e14,
}

_REFERENCE_STATE = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 1000.0], dtype=float)


# ---------------------------------------------------------------------------
# Orekit JVM initialisation (guarded — non-Orekit tests still run)
# ---------------------------------------------------------------------------

_OREKIT_AVAILABLE = False
try:
    import orekit

    orekit.initVM()
    from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

    setup_orekit_curdir()
    _OREKIT_AVAILABLE = True
except Exception:
    pass

requires_orekit = pytest.mark.skipif(not _OREKIT_AVAILABLE, reason="Orekit not available")


# ---------------------------------------------------------------------------
# 1) Registry / Factory integration (no Orekit required)
# ---------------------------------------------------------------------------


class TestGEqOEProviderRegistration:
    """Registry/factory integration tests — no Orekit dependency."""

    def test_provider_kind_is_geqoe(self) -> None:
        provider = GEqOEProvider()
        assert provider.kind == "geqoe"

    def test_provider_capabilities(self) -> None:
        provider = GEqOEProvider()
        cap = provider.capabilities
        assert isinstance(cap, CapabilityDescriptor)
        assert cap.supports_propagator is True
        assert cap.supports_builder is False
        assert cap.supports_stm is True
        assert cap.is_analytical is True
        assert cap.supports_custom_output is True

    def test_register_propagator_provider(self) -> None:
        registry = ProviderRegistry()
        provider = GEqOEProvider()
        registry.register_propagator_provider(provider)
        resolved = registry.get_propagator_provider("geqoe")
        assert resolved is provider

    def test_geqoe_in_available_propagator_kinds(self) -> None:
        registry = ProviderRegistry()
        registry.register_propagator_provider(GEqOEProvider())
        kinds = registry.available_propagator_kinds()
        assert "geqoe" in kinds

    def test_geqoe_not_in_builder_kinds(self) -> None:
        """GEqOE only registers as a propagator provider, not a builder provider."""
        registry = ProviderRegistry()
        registry.register_propagator_provider(GEqOEProvider())
        assert "geqoe" not in registry.available_builder_kinds()

    def test_factory_resolves_geqoe(self) -> None:
        factory = PropagatorFactory()
        factory.registry.register_propagator_provider(GEqOEProvider())
        spec = PropagatorSpec(kind="geqoe")
        resolved = factory.registry.get_propagator_provider(spec.kind)
        assert resolved.kind == "geqoe"

    def test_spec_accepts_plain_string_kind(self) -> None:
        """PropagatorSpec should accept plain string 'geqoe' without validation errors."""
        spec = PropagatorSpec(kind="geqoe")
        assert spec.kind == "geqoe"

    def test_spec_with_taylor_order(self) -> None:
        spec = PropagatorSpec(kind="geqoe", orekit_options={"taylor_order": 3})
        assert spec.orekit_options["taylor_order"] == 3

    def test_spec_with_mass(self) -> None:
        spec = PropagatorSpec(kind="geqoe", mass_kg=450.0)
        assert spec.mass_kg == 450.0


class TestGEqOEProviderRegistrationFunctions:
    """Test the registration helper functions."""

    def test_register_analytical_providers(self) -> None:
        from astrodyn_core.propagation.providers import register_analytical_providers

        registry = ProviderRegistry()
        register_analytical_providers(registry)
        assert "geqoe" in registry.available_propagator_kinds()

    def test_register_all_providers(self) -> None:
        from astrodyn_core.propagation.providers import register_all_providers
        from astrodyn_core.propagation.specs import PropagatorKind

        registry = ProviderRegistry()
        register_all_providers(registry)
        # Should have both Orekit-native and analytical kinds
        propagator_kinds = registry.available_propagator_kinds()
        assert "geqoe" in propagator_kinds
        # Orekit-native kinds use enum values (str representation)
        assert str(PropagatorKind.NUMERICAL) in propagator_kinds
        assert str(PropagatorKind.KEPLERIAN) in propagator_kinds

    def test_client_factory_includes_geqoe(self) -> None:
        from astrodyn_core.propagation.client import PropagationClient

        client = PropagationClient()
        factory = client.build_factory()
        kinds = factory.registry.available_propagator_kinds()
        assert "geqoe" in kinds


class TestGEqOEPropagatorWithoutOrekit:
    """Tests on the GEqOEPropagator class that don't require Orekit."""

    def test_propagator_stores_order(self) -> None:
        prop = GEqOEPropagator(
            initial_orbit=None,
            body_constants=_BODY_CONSTANTS,
            order=3,
        )
        assert prop.order == 3

    def test_propagator_stores_body_constants(self) -> None:
        prop = GEqOEPropagator(
            initial_orbit=None,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        bc = prop.body_constants
        assert bc["mu"] == _BODY_CONSTANTS["mu"]
        assert bc["j2"] == _BODY_CONSTANTS["j2"]
        assert bc["re"] == _BODY_CONSTANTS["re"]


# ---------------------------------------------------------------------------
# 2) Orekit adapter tests (skip if Orekit not available)
# ---------------------------------------------------------------------------


@requires_orekit
class TestGEqOEOrekitAdapter:
    """Orekit-dependent tests for the GEqOE propagator adapter."""

    def _make_initial_orbit(self):
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.frames import FramesFactory
        from org.orekit.orbits import CartesianOrbit
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.utils import PVCoordinates

        y0 = _REFERENCE_STATE
        pos = Vector3D(float(y0[0]), float(y0[1]), float(y0[2]))
        vel = Vector3D(float(y0[3]), float(y0[4]), float(y0[5]))
        pv = PVCoordinates(pos, vel)
        frame = FramesFactory.getGCRF()
        epoch = AbsoluteDate(2026, 1, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())
        mu = _BODY_CONSTANTS["mu"]
        return CartesianOrbit(pv, frame, epoch, mu), epoch

    def test_propagate_returns_spacecraft_state(self) -> None:
        from org.orekit.propagation import SpacecraftState

        orbit, epoch = self._make_initial_orbit()
        prop = GEqOEPropagator(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        target = epoch.shiftedBy(60.0)
        state = prop.propagate(target)
        assert isinstance(state, SpacecraftState)

    def test_propagate_pv_matches_numpy_engine(self) -> None:
        """Orekit output PV must match the pure-numpy ``taylor_cart_propagator``."""
        from astrodyn_core.propagation.geqoe.conversion import BodyConstants
        from astrodyn_core.propagation.geqoe.core import taylor_cart_propagator

        orbit, epoch = self._make_initial_orbit()
        prop = GEqOEPropagator(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        dt = 120.0
        target = epoch.shiftedBy(dt)
        state = prop.propagate(target)

        pv = state.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()
        orekit_y = np.array([
            float(pos.getX()), float(pos.getY()), float(pos.getZ()),
            float(vel.getX()), float(vel.getY()), float(vel.getZ()),
        ])

        bc = BodyConstants(
            j2=_BODY_CONSTANTS["j2"],
            re=_BODY_CONSTANTS["re"],
            mu=_BODY_CONSTANTS["mu"],
        )
        tspan = np.array([dt])
        y_np, _ = taylor_cart_propagator(tspan=tspan, y0=_REFERENCE_STATE, p=bc, order=4)

        np.testing.assert_allclose(orekit_y, y_np[0], rtol=1e-14, atol=1e-10)

    def test_get_native_state_returns_stm(self) -> None:
        orbit, epoch = self._make_initial_orbit()
        prop = GEqOEPropagator(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        target = epoch.shiftedBy(60.0)
        y, stm = prop.get_native_state(target)
        assert y.shape == (6,)
        assert stm.shape == (6, 6)
        # STM at t=0 would be identity; at t=60 it should be close but not identical
        assert not np.allclose(stm, np.eye(6))

    def test_propagate_at_epoch_zero_returns_initial(self) -> None:
        orbit, epoch = self._make_initial_orbit()
        prop = GEqOEPropagator(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        state = prop.propagate(epoch)
        pv = state.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()
        y = np.array([
            float(pos.getX()), float(pos.getY()), float(pos.getZ()),
            float(vel.getX()), float(vel.getY()), float(vel.getZ()),
        ])
        np.testing.assert_allclose(y, _REFERENCE_STATE, rtol=1e-13, atol=1e-10)

    def test_reset_initial_state(self) -> None:
        orbit, epoch = self._make_initial_orbit()
        prop = GEqOEPropagator(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        # Propagate to t=60, then reset to that state
        target1 = epoch.shiftedBy(60.0)
        state1 = prop.propagate(target1)
        prop.resetInitialState(state1)

        # Propagate 0s from reset => should get same state back
        state2 = prop.propagate(target1)
        pv1 = state1.getPVCoordinates()
        pv2 = state2.getPVCoordinates()
        np.testing.assert_allclose(
            [
                float(pv2.getPosition().getX()),
                float(pv2.getPosition().getY()),
                float(pv2.getPosition().getZ()),
            ],
            [
                float(pv1.getPosition().getX()),
                float(pv1.getPosition().getY()),
                float(pv1.getPosition().getZ()),
            ],
            rtol=1e-14,
        )

    def test_propagate_array(self) -> None:
        orbit, epoch = self._make_initial_orbit()
        prop = GEqOEPropagator(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
            order=4,
        )
        dt = np.array([0.0, 30.0, 60.0, 120.0])
        y_out, stm = prop.propagate_array(dt)
        assert y_out.shape == (4, 6)
        assert stm.shape == (6, 6, 4)
        # At t=0, state should match initial
        np.testing.assert_allclose(y_out[0], _REFERENCE_STATE, rtol=1e-13, atol=1e-10)

    def test_factory_build_propagator(self) -> None:
        """Full pipeline: factory -> provider -> propagator -> propagate."""
        from org.orekit.propagation import SpacecraftState

        orbit, epoch = self._make_initial_orbit()
        factory = PropagatorFactory()
        factory.registry.register_propagator_provider(GEqOEProvider())

        spec = PropagatorSpec(kind="geqoe", orekit_options={"taylor_order": 3})
        ctx = BuildContext(
            initial_orbit=orbit,
            body_constants=_BODY_CONSTANTS,
        )
        propagator = factory.build_propagator(spec, ctx)
        target = epoch.shiftedBy(60.0)
        state = propagator.propagate(target)
        assert isinstance(state, SpacecraftState)
