#!/usr/bin/env python
"""ASTRODYN-CORE demonstration script.

Exercises every major code path introduced through Phase 1 and Phase 1.1:

  1. Pure-Python spec validation (no Orekit needed)
  2. Registry and factory wiring (no Orekit needed)
  3. Keplerian propagation (Orekit, no force models)
  4. Numerical propagation with typed force specs (Orekit, full assembly)
  5. TLE propagation (Orekit, SGP4)

Run from the project root:
    python examples/demo_propagation.py
"""

from __future__ import annotations

import math
import sys
import textwrap

# ── Helpers ──────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 72


def header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Pure-Python validation (no Orekit / JVM required)
# ═══════════════════════════════════════════════════════════════════════════


def test_spec_validation() -> None:
    header("1 · Spec validation (pure Python)")

    from astrodyn_core import (
        AttitudeSpec,
        DragSpec,
        GravitySpec,
        IntegratorSpec,
        OceanTidesSpec,
        PropagatorKind,
        PropagatorSpec,
        RelativitySpec,
        SRPSpec,
        SolidTidesSpec,
        SpacecraftSpec,
        ThirdBodySpec,
        TLESpec,
    )

    # IntegratorSpec normalises kind
    ispec = IntegratorSpec(kind=" DP853 ", min_step=1e-3, max_step=300.0, position_tolerance=0.01)
    assert ispec.kind == "dp853"
    ok(f"IntegratorSpec normalised kind -> '{ispec.kind}'")

    # TLESpec validates line prefixes
    tle = TLESpec(
        line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002",
        line2="2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
    )
    ok(f"TLESpec created (NORAD {tle.line1[2:7].strip()})")

    try:
        TLESpec(line1="BAD", line2="2 ...")
        fail("Should have raised ValueError for bad TLE line1")
    except ValueError:
        ok("TLESpec rejects invalid line1 prefix")

    # PropagatorSpec validation
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        integrator=ispec,
        spacecraft=SpacecraftSpec(mass=500.0, drag_area=5.0),
        attitude=AttitudeSpec(mode="nadir"),
        force_specs=[
            GravitySpec(degree=20, order=20),
            DragSpec(atmosphere_model="nrlmsise00"),
            SRPSpec(),
            ThirdBodySpec(bodies=("sun", "moon", "venus")),
            RelativitySpec(),
            SolidTidesSpec(),
            OceanTidesSpec(degree=8, order=8),
        ],
    )
    ok(
        f"PropagatorSpec created: kind={spec.kind.value}, "
        f"{len(spec.force_specs)} force specs, "
        f"spacecraft mass={spec.spacecraft.mass} kg"
    )

    try:
        PropagatorSpec(kind=PropagatorKind.NUMERICAL)
        fail("Should have raised ValueError (missing integrator)")
    except ValueError:
        ok("PropagatorSpec rejects NUMERICAL without integrator")

    try:
        GravitySpec(degree=5, order=10)
        fail("Should have raised ValueError (order > degree)")
    except ValueError:
        ok("GravitySpec rejects order > degree")

    try:
        ThirdBodySpec(bodies=("pluto",))
        fail("Should have raised ValueError (unsupported body)")
    except ValueError:
        ok("ThirdBodySpec rejects unsupported body 'pluto'")

    try:
        AttitudeSpec(mode="invalid_mode")
        fail("Should have raised ValueError (bad mode)")
    except ValueError:
        ok("AttitudeSpec rejects invalid mode")

    # SpacecraftSpec box-wing normalises axis
    sc_bw = SpacecraftSpec(use_box_wing=True, solar_array_axis=(0.0, 3.0, 0.0))
    assert abs(sc_bw.solar_array_axis[1] - 1.0) < 1e-9
    ok(f"SpacecraftSpec normalised solar_array_axis -> {sc_bw.solar_array_axis}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Registry and factory wiring (no Orekit needed)
# ═══════════════════════════════════════════════════════════════════════════


def test_registry_factory() -> None:
    header("2 · Registry and factory wiring (pure Python)")

    from dataclasses import dataclass
    from astrodyn_core import (
        BuildContext,
        CapabilityDescriptor,
        PropagatorFactory,
        PropagatorKind,
        PropagatorSpec,
        IntegratorSpec,
        ProviderRegistry,
    )

    @dataclass(frozen=True)
    class StubProvider:
        kind: PropagatorKind = PropagatorKind.NUMERICAL
        capabilities: CapabilityDescriptor = CapabilityDescriptor()

        def build_builder(self, spec, ctx):
            return f"stub-builder({spec.kind.value})"

        def build_propagator(self, spec, ctx):
            return f"stub-propagator({spec.kind.value})"

    registry = ProviderRegistry()
    stub = StubProvider()
    registry.register_builder_provider(stub)
    registry.register_propagator_provider(stub)
    ok(f"Registered stub provider for {stub.kind.value}")

    factory = PropagatorFactory(registry=registry)

    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        integrator=IntegratorSpec(
            kind="dp853", min_step=0.001, max_step=300.0, position_tolerance=0.01
        ),
    )
    ctx = BuildContext()

    builder_result = factory.build_builder(spec, ctx)
    propagator_result = factory.build_propagator(spec, ctx)
    assert "stub-builder" in builder_result
    assert "stub-propagator" in propagator_result
    ok(f"Factory -> builder: {builder_result}")
    ok(f"Factory -> propagator: {propagator_result}")

    kinds = registry.available_builder_kinds()
    ok(f"Available builder kinds: {kinds}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Orekit-dependent tests
# ═══════════════════════════════════════════════════════════════════════════


def _init_orekit() -> bool:
    """Start the JVM and load orekit-data. Returns True on success."""
    try:
        import orekit

        orekit.initVM()
        from orekit.pyhelpers import setup_orekit_curdir

        setup_orekit_curdir()
        return True
    except Exception as exc:
        print(f"\n  Orekit init failed: {exc}")
        print("  Skipping Orekit-dependent tests.\n")
        return False


def _make_leo_orbit():
    """Create a simple LEO circular orbit for testing."""
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import Constants

    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc)
    eci = FramesFactory.getGCRF()
    mu = Constants.WGS84_EARTH_MU

    return KeplerianOrbit(
        6_878_137.0,  # a  (m)  ~500 km altitude
        0.0012,  # e
        math.radians(51.6),  # i
        math.radians(45.0),  # omega
        math.radians(120.0),  # RAAN
        math.radians(30.0),  # anomaly
        PositionAngleType.MEAN,
        eci,
        epoch,
        mu,
    )


def test_keplerian_propagation() -> None:
    header("3 · Keplerian propagation (Orekit)")

    from astrodyn_core import (
        BuildContext,
        PropagatorFactory,
        PropagatorKind,
        PropagatorSpec,
        ProviderRegistry,
        register_default_orekit_providers,
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)

    orbit = _make_leo_orbit()

    spec = PropagatorSpec(kind=PropagatorKind.KEPLERIAN)
    ctx = BuildContext(initial_orbit=orbit)

    builder = factory.build_builder(spec, ctx)
    ok(f"Keplerian builder type: {type(builder).__name__}")

    propagator = factory.build_propagator(spec, ctx)
    ok(f"Keplerian propagator type: {type(propagator).__name__}")

    # Propagate 1 orbit period
    from org.orekit.utils import Constants

    mu = Constants.WGS84_EARTH_MU
    a = orbit.getA()
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    final_state = propagator.propagate(orbit.getDate().shiftedBy(period))
    pv = final_state.getPVCoordinates()
    pos = pv.getPosition()
    ok(f"Propagated 1 period ({period:.1f} s)")
    ok(f"Final position: [{pos.getX():.1f}, {pos.getY():.1f}, {pos.getZ():.1f}] m")


def test_numerical_with_forces() -> None:
    header("4 · Numerical propagation with typed force specs (Orekit)")

    from astrodyn_core import (
        AttitudeSpec,
        BuildContext,
        DragSpec,
        GravitySpec,
        IntegratorSpec,
        PropagatorFactory,
        PropagatorKind,
        PropagatorSpec,
        ProviderRegistry,
        SRPSpec,
        SpacecraftSpec,
        ThirdBodySpec,
        register_default_orekit_providers,
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)

    orbit = _make_leo_orbit()

    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        integrator=IntegratorSpec(
            kind="dp853",
            min_step=1e-3,
            max_step=300.0,
            position_tolerance=1.0,
        ),
        spacecraft=SpacecraftSpec(
            mass=500.0,
            drag_area=5.0,
            drag_coeff=2.2,
            srp_area=5.0,
            srp_coeff=1.5,
        ),
        attitude=AttitudeSpec(mode="nadir"),
        force_specs=[
            GravitySpec(degree=8, order=8),
            DragSpec(atmosphere_model="harrispriester"),
            SRPSpec(),
            ThirdBodySpec(bodies=("sun", "moon")),
        ],
    )
    ctx = BuildContext(initial_orbit=orbit)

    builder = factory.build_builder(spec, ctx)
    ok(f"Numerical builder type: {type(builder).__name__}")

    # Inspect force models on the builder
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    forces = propagator.getAllForceModels()
    force_names = [f.getClass().getSimpleName() for f in forces]
    ok(f"Force models attached ({len(force_names)}): {force_names}")

    # Propagate 90 minutes
    duration = 90.0 * 60.0
    final_state = propagator.propagate(orbit.getDate().shiftedBy(duration))
    pv = final_state.getPVCoordinates()
    pos = pv.getPosition()
    vel = pv.getVelocity()
    ok(f"Propagated {duration:.0f} s (90 min)")
    ok(f"Final pos: [{pos.getX():.1f}, {pos.getY():.1f}, {pos.getZ():.1f}] m")
    ok(f"Final vel: [{vel.getX():.3f}, {vel.getY():.3f}, {vel.getZ():.3f}] m/s")


def test_tle_propagation() -> None:
    header("5 · TLE propagation (Orekit / SGP4)")

    from astrodyn_core import (
        BuildContext,
        PropagatorFactory,
        PropagatorKind,
        PropagatorSpec,
        ProviderRegistry,
        TLESpec,
        register_default_orekit_providers,
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)

    # ISS TLE (example epoch)
    tle = TLESpec(
        line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002",
        line2="2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
    )
    spec = PropagatorSpec(kind=PropagatorKind.TLE, tle=tle)
    ctx = BuildContext()

    propagator = factory.build_propagator(spec, ctx)
    ok(f"TLE propagator type: {type(propagator).__name__}")

    # Propagate 1 hour
    from org.orekit.propagation.analytical.tle import TLE as OrekitTLE

    orekit_tle = OrekitTLE(tle.line1, tle.line2)
    epoch = orekit_tle.getDate()

    final_state = propagator.propagate(epoch.shiftedBy(3600.0))
    pv = final_state.getPVCoordinates()
    pos = pv.getPosition()
    ok(f"Propagated 3600 s from TLE epoch")
    ok(f"Final pos: [{pos.getX():.1f}, {pos.getY():.1f}, {pos.getZ():.1f}] m")

    # Also test builder lane
    builder = factory.build_builder(spec, ctx)
    ok(f"TLE builder type: {type(builder).__name__}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — YAML configuration loader (pure Python + Orekit)
# ═══════════════════════════════════════════════════════════════════════════


def test_yaml_config_pure() -> None:
    header("6 · YAML config loader (pure Python)")

    from astrodyn_core import (
        get_propagation_model,
        list_propagation_models,
        load_dynamics_config,
        load_spacecraft_config,
        load_spacecraft_from_dict,
        SpacecraftSpec,
    )

    # List bundled presets
    presets = list_propagation_models()
    ok(f"Bundled presets: {presets}")

    # Load each preset and verify it produces a valid PropagatorSpec
    for name in presets:
        path = get_propagation_model(name)
        spec = load_dynamics_config(path)
        n_forces = len(spec.force_specs)
        has_integrator = spec.integrator is not None
        ok(
            f"  {name}: kind={spec.kind.value}, "
            f"integrator={'yes' if has_integrator else 'no'}, "
            f"forces={n_forces}"
        )

    # Load a preset and attach a spacecraft
    high_fid = load_dynamics_config(get_propagation_model("high_fidelity"))
    assert high_fid.spacecraft is None
    ok("high_fidelity loaded without spacecraft (as expected)")

    sc = SpacecraftSpec(mass=750.0, drag_area=8.0, drag_coeff=2.5)
    combined = high_fid.with_spacecraft(sc)
    assert combined.spacecraft is not None
    assert combined.spacecraft.mass == 750.0
    assert combined.force_specs == high_fid.force_specs  # forces unchanged
    ok(
        f"with_spacecraft() -> mass={combined.spacecraft.mass} kg, "
        f"forces preserved ({len(combined.force_specs)})"
    )

    # Load spacecraft from dict
    sc2 = load_spacecraft_from_dict({"mass": 300.0, "srp_area": 2.0})
    assert sc2.mass == 300.0
    ok(f"load_spacecraft_from_dict -> mass={sc2.mass}, srp_area={sc2.srp_area}")

    # Also test inline spacecraft YAML path loading
    import tempfile, yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"mass": 420.0, "drag_area": 3.5, "drag_coeff": 2.0}, f)
        sc_path = f.name

    spec_with_sc = load_dynamics_config(
        get_propagation_model("medium_fidelity"),
        spacecraft=sc_path,
    )
    assert spec_with_sc.spacecraft is not None
    assert spec_with_sc.spacecraft.mass == 420.0
    ok(f"load_dynamics_config(spacecraft=path) -> mass={spec_with_sc.spacecraft.mass}")

    import os

    os.unlink(sc_path)


def test_yaml_config_orekit() -> None:
    header("7 · YAML-configured numerical propagation (Orekit)")

    from astrodyn_core import (
        BuildContext,
        PropagatorFactory,
        ProviderRegistry,
        SpacecraftSpec,
        get_propagation_model,
        load_dynamics_config,
        register_default_orekit_providers,
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)

    orbit = _make_leo_orbit()

    # Load medium fidelity preset + custom spacecraft, build and propagate
    spec = load_dynamics_config(get_propagation_model("medium_fidelity"))
    spec = spec.with_spacecraft(SpacecraftSpec(mass=600.0, drag_area=6.0, srp_area=6.0))

    ctx = BuildContext(initial_orbit=orbit)
    builder = factory.build_builder(spec, ctx)
    ok(f"Builder from YAML preset: {type(builder).__name__}")

    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    forces = propagator.getAllForceModels()
    force_names = [f.getClass().getSimpleName() for f in forces]
    ok(f"Force models ({len(force_names)}): {force_names}")

    final_state = propagator.propagate(orbit.getDate().shiftedBy(3600.0))
    pos = final_state.getPVCoordinates().getPosition()
    ok(f"Propagated 3600 s from YAML config")
    ok(f"Final pos: [{pos.getX():.1f}, {pos.getY():.1f}, {pos.getZ():.1f}] m")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    print(SEPARATOR)
    print("  ASTRODYN-CORE  —  Functionality Demo")
    print(SEPARATOR)

    # Pure-Python tests (always run)
    test_spec_validation()
    test_registry_factory()
    test_yaml_config_pure()

    # Orekit-dependent tests
    if _init_orekit():
        test_keplerian_propagation()
        test_numerical_with_forces()
        test_tle_propagation()
        test_yaml_config_orekit()
    else:
        print("\n  Orekit not available — skipping Parts 3-5, 7.")

    header("All done")
    print()


if __name__ == "__main__":
    main()
