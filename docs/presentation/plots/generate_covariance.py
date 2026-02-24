#!/usr/bin/env python
"""Generate covariance growth plot for beamer presentation.

Run from project root:
    python docs/presentation/plots/generate_covariance.py
"""

import sys
import math
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
from examples._common import init_orekit, make_leo_orbit


def generate_covariance_plot():
    """Generate covariance growth plot showing position/velocity uncertainty evolution."""
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
        UncertaintySpec,
    )
    from astrodyn_core.uncertainty import setup_stm_propagator

    init_orekit()

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    out_dir = Path(__file__).parent

    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        spacecraft=SpacecraftSpec(mass=450.0, drag_area=5.0),
        integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-6),
        force_specs=[GravitySpec(degree=2, order=2)]
    )

    ctx = BuildContext(initial_orbit=orbit)
    builder = app.propagation.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    P0 = np.diag([1e4, 1e4, 1e4, 1e-2, 1e-2, 1e-2])

    period_s = 2.0 * math.pi * math.sqrt(orbit.getA() ** 3 / orbit.getMu())
    n_orbits = 32

    epoch_spec = OutputEpochSpec(
        start_epoch=app.state.from_orekit_date(epoch),
        end_epoch=app.state.from_orekit_date(epoch.shiftedBy(n_orbits * period_s)),
        step_seconds=600.0,
    )

    cov_spec = UncertaintySpec(method="stm", orbit_type="CARTESIAN")

    state_series, cov_series = app.uncertainty.propagate_with_covariance(
        propagator, P0, epoch_spec, spec=cov_spec, frame="GCRF"
    )

    times = []
    sig_pos = []
    sig_vel = []

    for rec in cov_series.records:
        cov = rec.to_numpy()
        times.append(rec.epoch)
        sig_pos.append(np.sqrt(np.trace(cov[:3, :3]) / 3.0))
        sig_vel.append(np.sqrt(np.trace(cov[3:, 3:]) / 3.0))

    time_hours = np.arange(len(times)) * 600.0 / 3600.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_hours, np.array(sig_pos) / 1e3, 'b-', linewidth=2, label='Position σ')
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Position Uncertainty (km)', fontsize=12)
    ax1.set_title('Position Uncertainty Growth (3D RMS)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(time_hours, sig_vel, 'r-', linewidth=2, label='Velocity σ')
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Velocity Uncertainty (m/s)', fontsize=12)
    ax2.set_title('Velocity Uncertainty Growth', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "covariance_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'covariance_growth.png'}")


def main():
    print("=" * 60)
    print("Generating covariance growth plot")
    print("=" * 60)
    print()
    generate_covariance_plot()
    print()
    print("Plot generated successfully!")


if __name__ == "__main__":
    main()
