#!/usr/bin/env python
"""Generate plots for ASTRODYN-CORE beamer presentation.

This script creates the plots referenced in the beamer presentation:
1. fidelity_comparison.png - Multi-fidelity propagation comparison
2. orbital_elements.png - Orbital elements over time

Run from project root:
    python docs/presentation/plots/generate_plots.py
"""

import sys
import math
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np

from examples._common import init_orekit, make_generated_dir, make_leo_orbit


def plot_fidelity_comparison():
    """Generate multi-fidelity comparison plot."""
    import matplotlib.pyplot as plt
    from astrodyn_core import (
        AstrodynClient,
        BuildContext,
        GravitySpec,
        IntegratorSpec,
        OutputEpochSpec,
        PropagatorKind,
        PropagatorSpec,
        SpacecraftSpec,
    )

    init_orekit()
    from org.orekit.orbits import KeplerianOrbit

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    period_s = 2.0 * math.pi * math.sqrt(orbit.getA() ** 3 / orbit.getMu())
    n_orbits = 4
    end_date = epoch.shiftedBy(n_orbits * period_s)

    epoch_spec = OutputEpochSpec(
        start_epoch=app.state.from_orekit_date(epoch),
        end_epoch=app.state.from_orekit_date(end_date),
        step_seconds=60.0,
    )

    configs = {
        "Keplerian": PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0),
        "DSST": PropagatorSpec(
            kind=PropagatorKind.DSST,
            spacecraft=SpacecraftSpec(mass=450.0, drag_area=5.0, srp_area=5.0),
            integrator=IntegratorSpec(kind="dp853", min_step=1e-3, max_step=300.0, position_tolerance=1e-3),
            dsst_propagation_type="OSCULATING",
            dsst_state_type="OSCULATING",
            force_specs=[GravitySpec(degree=8, order=8)]
        ),
        "Numerical": PropagatorSpec(
            kind=PropagatorKind.NUMERICAL,
            spacecraft=SpacecraftSpec(mass=450.0, drag_area=5.0, srp_area=5.0),
            integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-3),
            force_specs=[GravitySpec(degree=8, order=8)]
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Multi-Fidelity Propagation Comparison", fontsize=14, fontweight='bold')

    for name, spec in configs.items():
        ctx = BuildContext(initial_orbit=orbit)
        builder = app.propagation.build_builder(spec, ctx)
        propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

        times = []
        smas = []
        eccs = []
        incs = []

        current_date = epoch
        while current_date.isBefore(end_date) or current_date.isEqualTo(end_date):
            state = propagator.propagate(current_date)
            orb = state.getOrbit()
            times.append((current_date.durationFrom(epoch)) / 3600.0)
            smas.append(orb.getA() / 1e3)
            eccs.append(orb.getE())
            incs.append(math.degrees(orb.getI()))

            current_date = current_date.shiftedBy(60.0)

        axes[0, 0].plot(times, smas, label=name, linewidth=2)
        axes[0, 1].plot(times, eccs, label=name, linewidth=2)
        axes[1, 0].plot(times, incs, label=name, linewidth=2)

    axes[0, 0].set_xlabel("Time (hours)")
    axes[0, 0].set_ylabel("Semi-major Axis (km)")
    axes[0, 0].set_title("SMA Evolution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Time (hours)")
    axes[0, 1].set_ylabel("Eccentricity")
    axes[0, 1].set_title("Eccentricity")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Time (hours)")
    axes[1, 0].set_ylabel("Inclination (deg)")
    axes[1, 0].set_title("Inclination")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    ctx = BuildContext(initial_orbit=orbit)
    builder = app.propagation.build_builder(PropagatorSpec(kind=PropagatorKind.NUMERICAL,
        spacecraft=SpacecraftSpec(mass=450.0),
        integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-3),
        force_specs=[GravitySpec(degree=2, order=2)]), ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    times = []
    raans = []
    args = []
    for current_date in [epoch.shiftedBy(float(t)) for t in np.arange(0, n_orbits * period_s, 60)]:
        state = propagator.propagate(current_date)
        orb = KeplerianOrbit.cast_(state.getOrbit())
        times.append((current_date.durationFrom(epoch)) / 3600.0)
        raans.append(math.degrees(orb.getRightAscensionOfAscendingNode()))
        args.append(math.degrees(orb.getPerigeeArgument()))

    axes[1, 1].plot(times, raans, label="RAAN", linewidth=2)
    axes[1, 1].plot(times, args, label="Arg. Perigee", linewidth=2)
    axes[1, 1].set_xlabel("Time (hours)")
    axes[1, 1].set_ylabel("Angle (deg)")
    axes[1, 1].set_title("RAAN & Argument of Perigee (J2 precession)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fidelity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'fidelity_comparison.png'}")


def plot_orbital_elements():
    """Generate orbital elements plot for numerical propagation."""
    import matplotlib.pyplot as plt
    from astrodyn_core import (
        AstrodynClient,
        BuildContext,
        GravitySpec,
        IntegratorSpec,
        OutputEpochSpec,
        PropagatorKind,
        PropagatorSpec,
        SpacecraftSpec,
    )

    init_orekit()
    from org.orekit.orbits import KeplerianOrbit

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    out_dir = Path(__file__).parent

    period_s = 2.0 * math.pi * math.sqrt(orbit.getA() ** 3 / orbit.getMu())
    n_orbits = 10

    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        spacecraft=SpacecraftSpec(mass=450.0),
        integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-3),
        force_specs=[GravitySpec(degree=2, order=2)]
    )

    ctx = BuildContext(initial_orbit=orbit)
    builder = app.propagation.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    times = []
    smas = []
    eccs = []
    incs = []
    raans = []
    args = []
    anomalies = []

    end_date = epoch.shiftedBy(n_orbits * period_s)
    current_date = epoch

    while current_date.isBefore(end_date) or current_date.isEqualTo(end_date):
        state = propagator.propagate(current_date)
        orb = KeplerianOrbit.cast_(state.getOrbit())
        times.append((current_date.durationFrom(epoch)) / 3600.0)
        smas.append(orb.getA() / 1e3)
        eccs.append(orb.getE())
        incs.append(math.degrees(orb.getI()))
        raans.append(math.degrees(orb.getRightAscensionOfAscendingNode()))
        args.append(math.degrees(orb.getPerigeeArgument()))
        anomalies.append(math.degrees(orb.getTrueAnomaly()))

        current_date = current_date.shiftedBy(60.0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Orbital Elements Evolution (Numerical + J2)", fontsize=14, fontweight='bold')

    axes[0, 0].plot(times, smas, 'b-', linewidth=2)
    axes[0, 0].set_xlabel("Time (hours)")
    axes[0, 0].set_ylabel("SMA (km)")
    axes[0, 0].set_title("Semi-major Axis")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, eccs, 'r-', linewidth=2)
    axes[0, 1].set_xlabel("Time (hours)")
    axes[0, 1].set_ylabel("Eccentricity")
    axes[0, 1].set_title("Eccentricity")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, incs, 'g-', linewidth=2)
    axes[1, 0].set_xlabel("Time (hours)")
    axes[1, 0].set_ylabel("Inclination (deg)")
    axes[1, 0].set_title("Inclination")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, raans, 'm-', linewidth=2)
    axes[1, 1].set_xlabel("Time (hours)")
    axes[1, 1].set_ylabel("RAAN (deg)")
    axes[1, 1].set_title("Right Ascension of Ascending Node (J2 precession)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(times, args, 'c-', linewidth=2)
    axes[2, 0].set_xlabel("Time (hours)")
    axes[2, 0].set_ylabel("Arg. Perigee (deg)")
    axes[2, 0].set_title("Argument of Perigee (J2 precession)")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(times, anomalies, 'orange', linewidth=2)
    axes[2, 1].set_xlabel("Time (hours)")
    axes[2, 1].set_ylabel("True Anomaly (deg)")
    axes[2, 1].set_title("True Anomaly (periodic)")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "orbital_elements.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'orbital_elements.png'}")


def main():
    print("=" * 60)
    print("Generating plots for ASTRODYN-CORE presentation")
    print("=" * 60)
    print()
    plot_fidelity_comparison()
    plot_orbital_elements()
    print()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
