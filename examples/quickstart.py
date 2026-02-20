#!/usr/bin/env python
"""Quickstart entry point for ASTRODYN-CORE capabilities.

Covers the most common first-day workflows in one script:
1. Config/spec discovery and validation
2. Keplerian propagation
3. Numerical propagation with YAML dynamics + spacecraft presets
4. DSST propagation
5. TLE propagation
6. TLE cache resolution (NORAD + epoch -> TLESpec)
7. Trajectory export + orbital-elements plot

Run from project root:
    python examples/quickstart.py --mode all
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _common import (
    build_factory,
    init_orekit,
    make_generated_dir,
    make_leo_orbit,
)


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_basics() -> None:
    """Show model/config surface area without requiring Orekit."""
    _header("Quickstart · Basics")
    from astrodyn_core import (
        IntegratorSpec,
        PropagatorKind,
        PropagatorSpec,
        SpacecraftSpec,
        get_propagation_model,
        get_spacecraft_model,
        list_propagation_models,
        list_spacecraft_models,
        load_dynamics_config,
        load_spacecraft_config,
    )

    propagation_models = list_propagation_models()
    spacecraft_models = list_spacecraft_models()
    print(f"Bundled propagation models ({len(propagation_models)}): {propagation_models}")
    print(f"Bundled spacecraft models ({len(spacecraft_models)}): {spacecraft_models}")

    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        integrator=IntegratorSpec(kind="dp853", min_step=1e-3, max_step=300.0, position_tolerance=1.0),
        spacecraft=SpacecraftSpec(mass=500.0, drag_area=4.0),
    )
    print(f"Validated PropagatorSpec kind={spec.kind.value}, integrator={spec.integrator.kind}")

    high_fid = load_dynamics_config(get_propagation_model("high_fidelity"))
    leo_sc = load_spacecraft_config(get_spacecraft_model("leo_smallsat"))
    print(
        "Loaded YAML presets: "
        f"dynamics={high_fid.kind.value}, spacecraft mass={leo_sc.mass} kg"
    )


def run_keplerian() -> None:
    _header("Quickstart · Keplerian")
    init_orekit()

    from astrodyn_core import BuildContext, PropagatorKind, PropagatorSpec

    orbit, epoch, frame = make_leo_orbit()
    factory = build_factory()
    spec = PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0)

    builder = factory.build_builder(spec, BuildContext(initial_orbit=orbit))
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    target = epoch.shiftedBy(3600.0)
    state = propagator.propagate(target)
    pos = state.getPVCoordinates(frame).getPosition()
    print(f"Builder: {builder.getClass().getSimpleName()}")
    print(f"Propagator: {propagator.getClass().getSimpleName()}")
    print(f"Position after 1h (km): ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})")


def run_numerical() -> None:
    _header("Quickstart · Numerical")
    init_orekit()

    from astrodyn_core import (
        BuildContext,
        SpacecraftSpec,
        get_propagation_model,
        load_dynamics_config,
    )

    orbit, epoch, frame = make_leo_orbit()
    spec = load_dynamics_config(get_propagation_model("medium_fidelity"))
    spec = spec.with_spacecraft(SpacecraftSpec(mass=600.0, drag_area=6.0, srp_area=6.0))

    factory = build_factory()
    builder = factory.build_builder(spec, BuildContext(initial_orbit=orbit))
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    forces = [f.getClass().getSimpleName() for f in propagator.getAllForceModels()]
    state = propagator.propagate(epoch.shiftedBy(5400.0))
    pos = state.getPVCoordinates(frame).getPosition()
    vel = state.getPVCoordinates(frame).getVelocity()

    print(f"Force models ({len(forces)}): {forces}")
    print(f"Position after 90 min (m): [{pos.getX():.1f}, {pos.getY():.1f}, {pos.getZ():.1f}]")
    print(f"Velocity after 90 min (m/s): [{vel.getX():.3f}, {vel.getY():.3f}, {vel.getZ():.3f}]")


def run_dsst() -> None:
    _header("Quickstart · DSST")
    init_orekit()

    from astrodyn_core import BuildContext, IntegratorSpec, PropagatorKind, PropagatorSpec

    orbit, epoch, frame = make_leo_orbit()
    factory = build_factory()
    spec = PropagatorSpec(
        kind=PropagatorKind.DSST,
        mass_kg=550.0,
        integrator=IntegratorSpec(kind="dp853", min_step=1.0e-3, max_step=300.0, position_tolerance=10.0),
        dsst_propagation_type="MEAN",
        dsst_state_type="OSCULATING",
    )

    builder = factory.build_builder(spec, BuildContext(initial_orbit=orbit))
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    state = propagator.propagate(epoch.shiftedBy(2.0 * 3600.0))
    pos = state.getPVCoordinates(frame).getPosition()
    print(f"Builder: {builder.getClass().getSimpleName()}")
    print(f"Propagator: {propagator.getClass().getSimpleName()}")
    print(f"Position after 2h (km): ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})")


def run_tle() -> None:
    _header("Quickstart · TLE")
    init_orekit()

    from org.orekit.frames import FramesFactory
    from org.orekit.propagation.analytical.tle import TLE as OrekitTLE

    from astrodyn_core import BuildContext, PropagatorKind, PropagatorSpec, TLESpec

    tle = TLESpec(
        line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002",
        line2="2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
    )

    factory = build_factory()
    propagator = factory.build_propagator(PropagatorSpec(kind=PropagatorKind.TLE, tle=tle), BuildContext())
    tle_epoch = OrekitTLE(tle.line1, tle.line2).getDate()
    state = propagator.propagate(tle_epoch.shiftedBy(45.0 * 60.0))
    pos = state.getPVCoordinates(FramesFactory.getGCRF()).getPosition()
    orbit = state.getOrbit()

    print(f"Propagator: {propagator.getClass().getSimpleName()}")
    print(f"a={orbit.getA()/1e3:.2f} km, e={orbit.getE():.6f}, i={math.degrees(orbit.getI()):.3f} deg")
    print(f"Position after 45 min (km): ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})")


def run_tle_resolve() -> None:
    _header("Quickstart · TLE Resolve (Download + Cache)")
    init_orekit()

    from configparser import ConfigParser
    from datetime import datetime, timezone

    from org.orekit.frames import FramesFactory
    from org.orekit.propagation.analytical.tle import TLE as OrekitTLE
    from spacetrack import SpaceTrackClient

    from astrodyn_core import (
        BuildContext,
        PropagatorKind,
        PropagatorSpec,
        TLEQuery,
        resolve_tle_spec,
    )

    out_dir = make_generated_dir()
    tle_cache_dir = out_dir / "tle_cache"

    repo_root = Path(__file__).resolve().parents[1]
    secrets_path = repo_root / "secrets.ini"
    if not secrets_path.exists():
        raise RuntimeError(f"Missing credentials file: {secrets_path}")

    cfg = ConfigParser()
    cfg.read(secrets_path)
    identity = cfg.get("credentials", "spacetrack_identity", fallback="").strip()
    password = cfg.get("credentials", "spacetrack_password", fallback="").strip()
    if not identity or not password:
        raise RuntimeError(
            "secrets.ini is missing credentials.spacetrack_identity or credentials.spacetrack_password."
        )

    space_track_client = SpaceTrackClient(identity=identity, password=password)

    # Download-backed resolution for ISS/NORAD 25544.
    norad_id = 25544
    target_epoch = datetime.now(timezone.utc)

    query = TLEQuery(
        norad_id=norad_id,
        target_epoch=target_epoch,
        base_dir=tle_cache_dir,
        allow_download=True,
    )
    tle_spec = resolve_tle_spec(query, space_track_client=space_track_client)

    factory = build_factory()
    propagator = factory.build_propagator(
        PropagatorSpec(kind=PropagatorKind.TLE, tle=tle_spec),
        BuildContext(),
    )

    tle_epoch = OrekitTLE(tle_spec.line1, tle_spec.line2).getDate()
    state = propagator.propagate(tle_epoch.shiftedBy(30.0 * 60.0))
    pos = state.getPVCoordinates(FramesFactory.getGCRF()).getPosition()

    print(f"Target epoch: {target_epoch.isoformat()}")
    print(f"TLE cache directory: {tle_cache_dir}")
    print(f"Resolved TLE line1: {tle_spec.line1}")
    print(f"Position after 30 min (km): ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})")


def run_plot() -> None:
    _header("Quickstart · Export + Plot")
    init_orekit()

    from astrodyn_core import BuildContext, OutputEpochSpec, PropagatorKind, PropagatorSpec, StateFileClient
    from astrodyn_core.states.orekit import from_orekit_date

    orbit, epoch, _frame = make_leo_orbit()
    factory = build_factory()
    builder = factory.build_builder(
        PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0),
        BuildContext(initial_orbit=orbit),
    )
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    period_s = 2.0 * math.pi * math.sqrt(orbit.getA() ** 3 / orbit.getMu())
    end_epoch = from_orekit_date(epoch.shiftedBy(3.0 * period_s))
    epoch_spec = OutputEpochSpec(
        start_epoch=from_orekit_date(epoch),
        end_epoch=end_epoch,
        step_seconds=60.0,
    )

    out_dir = make_generated_dir()
    series_path = out_dir / "quickstart_orbit_series.yaml"
    plot_path = out_dir / "quickstart_orbit_elements.png"

    client = StateFileClient()
    client.export_trajectory_from_propagator(
        propagator,
        epoch_spec,
        series_path,
        series_name="quickstart-orbit",
        representation="keplerian",
        frame="GCRF",
    )
    series = client.load_state_series(series_path)
    client.plot_orbital_elements(series, plot_path, title="Quickstart: Orbital Elements")
    print(f"Saved trajectory: {series_path}")
    print(f"Saved plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRODYN-CORE quickstart examples")
    parser.add_argument(
        "--mode",
        choices=("all", "basics", "keplerian", "numerical", "dsst", "tle", "tle_resolve", "plot"),
        default="all",
        help="Choose one workflow or run all (default).",
    )
    args = parser.parse_args()

    steps = {
        "basics": run_basics,
        "keplerian": run_keplerian,
        "numerical": run_numerical,
        "dsst": run_dsst,
        "tle": run_tle,
        "tle_resolve": run_tle_resolve,
        "plot": run_plot,
    }

    if args.mode == "all":
        for key in ("basics", "keplerian", "numerical", "dsst", "tle", "tle_resolve", "plot"):
            steps[key]()
    else:
        steps[args.mode]()


if __name__ == "__main__":
    main()
