"""Tests for DSST force model assembly.

Covers both pure-Python unit tests (no JVM) and Orekit integration tests.
"""

import warnings

import pytest

from astrodyn_core.propagation.forces import (
    GravitySpec,
    DragSpec,
    OceanTidesSpec,
    RelativitySpec,
    SolidTidesSpec,
    SRPSpec,
    ThirdBodySpec,
)
from astrodyn_core.propagation.spacecraft import SpacecraftSpec

# ---------------------------------------------------------------------------
# Pure-Python tests (no Orekit / JVM required)
# ---------------------------------------------------------------------------


def test_assemble_dsst_force_models_importable_from_package() -> None:
    """assemble_dsst_force_models is accessible from the propagation package."""
    from astrodyn_core.propagation import assemble_dsst_force_models

    assert callable(assemble_dsst_force_models)


def test_unsupported_specs_emit_warnings() -> None:
    """RelativitySpec, SolidTidesSpec, OceanTidesSpec should warn and be skipped."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    unsupported = [RelativitySpec(), SolidTidesSpec(), OceanTidesSpec()]
    sc = SpacecraftSpec()

    # Use a dummy orbit object with getMu method
    class _FakeOrbit:
        def getMu(self):
            return 3.986004415e14

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = assemble_dsst_force_models(unsupported, sc, _FakeOrbit())

    assert result == [], "Unsupported specs should produce no DSST models"
    assert len(caught) == 3, f"Expected 3 warnings, got {len(caught)}"
    for w in caught:
        assert "not supported for DSST" in str(w.message)


def test_unknown_spec_type_raises_type_error() -> None:
    """A completely unknown spec type should raise TypeError."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    class _BogusSpec:
        pass

    class _FakeOrbit:
        def getMu(self):
            return 3.986004415e14

    with pytest.raises(TypeError, match="Unknown force spec type"):
        assemble_dsst_force_models([_BogusSpec()], SpacecraftSpec(), _FakeOrbit())


def test_point_mass_gravity_returns_empty() -> None:
    """GravitySpec(degree=0, order=0) should produce an empty list (point mass)."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    class _FakeOrbit:
        def getMu(self):
            return 3.986004415e14

    result = assemble_dsst_force_models(
        [GravitySpec(degree=0, order=0)], SpacecraftSpec(), _FakeOrbit()
    )
    assert result == []


# ---------------------------------------------------------------------------
# Orekit integration tests — require JVM + Orekit data
# ---------------------------------------------------------------------------

orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()


def _make_leo_orbit():
    """Create a LEO circular orbit for integration tests."""
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import Constants

    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2026, 2, 19, 0, 0, 0.0, utc)
    mu = Constants.WGS84_EARTH_MU
    gcrf = FramesFactory.getGCRF()
    return KeplerianOrbit(
        6_878_137.0,  # a (m) — ~500 km altitude
        0.001,  # e
        0.9,  # i (rad)
        0.0,  # omega
        0.0,  # RAAN
        0.0,  # anomaly
        PositionAngleType.MEAN,
        gcrf,
        epoch,
        mu,
    )


def test_gravity_spec_produces_dsst_zonal_and_tesseral() -> None:
    """GravitySpec with degree>0 and order>0 should produce DSSTZonal + DSSTTesseral."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTTesseral, DSSTZonal

    orbit = _make_leo_orbit()
    result = assemble_dsst_force_models(
        [GravitySpec(degree=4, order=4)], SpacecraftSpec(), orbit
    )

    assert len(result) == 2
    type_names = {type(m).__name__ for m in result}
    assert "DSSTZonal" in type_names
    assert "DSSTTesseral" in type_names


def test_gravity_zonal_only_when_order_zero() -> None:
    """GravitySpec with order=0 (but degree>0) should produce only DSSTZonal."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTZonal

    orbit = _make_leo_orbit()
    result = assemble_dsst_force_models(
        [GravitySpec(degree=4, order=0)], SpacecraftSpec(), orbit
    )

    assert len(result) == 1
    assert type(result[0]).__name__ == "DSSTZonal"


def test_third_body_produces_dsst_third_body() -> None:
    """ThirdBodySpec should produce DSSTThirdBody instances."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    orbit = _make_leo_orbit()
    result = assemble_dsst_force_models(
        [ThirdBodySpec(bodies=("sun", "moon"))], SpacecraftSpec(), orbit
    )

    assert len(result) == 2
    for m in result:
        assert type(m).__name__ == "DSSTThirdBody"


def test_drag_spec_produces_dsst_atmospheric_drag() -> None:
    """DragSpec should produce a DSSTAtmosphericDrag instance."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    orbit = _make_leo_orbit()
    result = assemble_dsst_force_models(
        [DragSpec(atmosphere_model="harrispriester")], SpacecraftSpec(), orbit
    )

    assert len(result) == 1
    assert type(result[0]).__name__ == "DSSTAtmosphericDrag"


def test_srp_spec_produces_dsst_solar_radiation_pressure() -> None:
    """SRPSpec should produce a DSSTSolarRadiationPressure instance."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    orbit = _make_leo_orbit()
    result = assemble_dsst_force_models(
        [SRPSpec()], SpacecraftSpec(), orbit
    )

    assert len(result) == 1
    assert type(result[0]).__name__ == "DSSTSolarRadiationPressure"


def test_mixed_specs_correct_count() -> None:
    """Multiple specs combined should produce the expected total count."""
    from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

    orbit = _make_leo_orbit()
    specs = [
        GravitySpec(degree=4, order=4),        # -> 2 (zonal + tesseral)
        ThirdBodySpec(bodies=("sun", "moon")),  # -> 2
        DragSpec(atmosphere_model="harrispriester"),  # -> 1
        SRPSpec(),                              # -> 1
        RelativitySpec(),                       # -> 0 (warn + skip)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = assemble_dsst_force_models(specs, SpacecraftSpec(), orbit)

    assert len(result) == 6
    assert len(caught) == 1  # only RelativitySpec warns


def test_dsst_provider_builds_propagator_with_force_specs() -> None:
    """DSSTOrekitProvider should build a working propagator when force specs are given."""
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    from astrodyn_core.propagation.interfaces import BuildContext
    from astrodyn_core.propagation.providers.orekit_native import DSSTOrekitProvider
    from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec

    orbit = _make_leo_orbit()
    spec = PropagatorSpec(
        kind=PropagatorKind.DSST,
        integrator=IntegratorSpec(
            kind="dp853", min_step=0.1, max_step=300.0, position_tolerance=10.0
        ),
        force_specs=[GravitySpec(degree=4, order=4)],
    )
    ctx = BuildContext(initial_orbit=orbit)

    propagator = DSSTOrekitProvider().build_propagator(spec, ctx)

    utc = TimeScalesFactory.getUTC()
    end = AbsoluteDate(2026, 2, 19, 1, 0, 0.0, utc)  # +1 hour
    state = propagator.propagate(end)

    pos = state.getPVCoordinates().getPosition()
    # Sanity: position magnitude should be in LEO range (6000-8000 km)
    r_km = pos.getNorm() / 1000.0
    assert 6000.0 < r_km < 8000.0, f"Position {r_km:.1f} km not in LEO range"


def test_dsst_with_gravity_differs_from_keplerian() -> None:
    """DSST with gravity perturbations should differ from pure Keplerian propagation."""
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    from astrodyn_core.propagation.interfaces import BuildContext
    from astrodyn_core.propagation.providers.orekit_native import DSSTOrekitProvider
    from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec

    orbit = _make_leo_orbit()

    # Keplerian reference
    kep = KeplerianPropagator(orbit)
    utc = TimeScalesFactory.getUTC()
    end = AbsoluteDate(2026, 2, 19, 6, 0, 0.0, utc)  # +6 hours
    kep_state = kep.propagate(end)
    kep_pos = kep_state.getPVCoordinates().getPosition()

    # DSST with gravity
    spec = PropagatorSpec(
        kind=PropagatorKind.DSST,
        integrator=IntegratorSpec(
            kind="dp853", min_step=0.1, max_step=300.0, position_tolerance=10.0
        ),
        force_specs=[GravitySpec(degree=8, order=8)],
    )
    ctx = BuildContext(initial_orbit=orbit)
    dsst_prop = DSSTOrekitProvider().build_propagator(spec, ctx)
    dsst_state = dsst_prop.propagate(end)
    dsst_pos = dsst_state.getPVCoordinates().getPosition()

    # The difference should be significant (>> 0)
    diff_km = kep_pos.subtract(dsst_pos).getNorm() / 1000.0
    assert diff_km > 0.1, (
        f"DSST with J2-J8 should differ from Keplerian by > 100m over 6h, "
        f"got {diff_km:.4f} km"
    )
