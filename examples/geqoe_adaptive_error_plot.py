#!/usr/bin/env python
"""Plot position error of adaptive GEqOE vs numerical GEqOE reference.

Produces one figure per ``max_step`` value, each with order 2/3/4 traces.
Figures are saved to ``examples/generated/geqoe/``.

Run from project root:
    python examples/geqoe_adaptive_error_plot.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from _common import init_orekit, make_generated_dir, make_leo_orbit


def main() -> None:
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, _, _ = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)

    a = float(orbit.getA())
    mu = float(orbit.getMu())
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)

    n_orbits = 10
    dt_grid = np.linspace(0, n_orbits * period, 2000)
    t_orbits = dt_grid / period

    # Numerical GEqOE reference
    ref_prop = app.propagation.build_propagator(
        PropagatorSpec(
            kind="geqoe-numerical",
            orekit_options={"rtol": 1e-13, "atol": 1e-13},
        ),
        ctx,
    )
    ref, _ = ref_prop.propagate_array(dt_grid)

    # Pre-compute numerical integrator traces at various tolerances
    num_tols = [1e-6, 1e-8, 1e-10]
    num_results = {}  # tol -> (pos_err, nfev)
    for tol in num_tols:
        num_prop = app.propagation.build_propagator(
            PropagatorSpec(
                kind="geqoe-numerical",
                orekit_options={"rtol": tol, "atol": tol},
            ),
            ctx,
        )
        y_num, _ = num_prop.propagate_array(dt_grid)
        pos_err = np.linalg.norm(y_num[:, :3] - ref[:, :3], axis=1)
        num_results[tol] = (pos_err, num_prop.nfev)

    out_dir = make_generated_dir() / "geqoe"
    out_dir.mkdir(exist_ok=True)

    max_steps = [600.0, 400.0, 300.0, 120.0, 60.0, 30.0]
    orders = [2, 3, 4]

    # --- Fixed max_step figures ---
    for max_step in max_steps:
        fig, ax = plt.subplots(figsize=(9, 4))

        for order in orders:
            prop = app.propagation.build_propagator(
                PropagatorSpec(
                    kind="geqoe-adaptive",
                    orekit_options={
                        "taylor_order": order,
                        "backend": "cpp",
                        "max_step": max_step,
                    },
                ),
                ctx,
            )
            y, _ = prop.propagate_array(dt_grid)
            pos_err = np.linalg.norm(y[:, :3] - ref[:, :3], axis=1)
            n_ckpts = prop.num_checkpoints
            ax.plot(t_orbits, pos_err, label=f"Order {order} ({n_ckpts} ckpts)")

        for ntol, (nerr, nfev) in num_results.items():
            ax.plot(t_orbits, nerr, "--", alpha=0.6,
                    label=f"Numerical (tol={ntol:.0e}, {nfev} evals)")

        ax.set_xlabel("Orbits")
        ax.set_ylabel("Position error [m]")
        ax.set_title(f"Adaptive GEqOE vs numerical GEqOE — max_step = {max_step:.0f} s")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_orbits)

        fname = out_dir / f"error_max_step_{max_step:.0f}s.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    # --- pos_tol mode figures (no max_step cap) ---
    pos_tols = [10.0, 1.0, 0.1, 0.01]

    for tol in pos_tols:
        fig, ax = plt.subplots(figsize=(9, 4))

        for order in orders:
            prop = app.propagation.build_propagator(
                PropagatorSpec(
                    kind="geqoe-adaptive",
                    orekit_options={
                        "taylor_order": order,
                        "backend": "cpp",
                        "pos_tol": tol,
                    },
                ),
                ctx,
            )
            y, _ = prop.propagate_array(dt_grid)
            pos_err = np.linalg.norm(y[:, :3] - ref[:, :3], axis=1)
            n_ckpts = prop.num_checkpoints
            ax.plot(t_orbits, pos_err, label=f"Order {order} ({n_ckpts} ckpts)")

        for ntol, (nerr, nfev) in num_results.items():
            ax.plot(t_orbits, nerr, "--", alpha=0.6,
                    label=f"Numerical (tol={ntol:.0e}, {nfev} evals)")

        ax.set_xlabel("Orbits")
        ax.set_ylabel("Position error [m]")
        ax.set_title(f"Adaptive GEqOE vs numerical GEqOE — pos_tol = {tol} m")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_orbits)

        fname = out_dir / f"error_pos_tol_{tol}m.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


if __name__ == "__main__":
    main()
