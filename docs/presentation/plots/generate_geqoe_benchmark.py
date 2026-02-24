#!/usr/bin/env python
"""Generate GEqOE benchmark plot for beamer presentation.

Run from project root:
    python docs/presentation/plots/generate_geqoe_benchmark.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
from examples._common import init_orekit, make_leo_orbit


def generate_geqoe_benchmark_plot():
    """Generate benchmark comparison plot for GEqOE propagator."""
    import matplotlib.pyplot as plt
    from astrodyn_core.propagation.geqoe.conversion import BodyConstants
    from astrodyn_core.propagation.geqoe.core import taylor_cart_propagator
    from astrodyn_core.propagation.providers.geqoe import GEqOEPropagator

    init_orekit()

    orbit, epoch, frame = make_leo_orbit()
    out_dir = Path(__file__).parent

    rng = np.random.default_rng(1)
    N_EPOCHS = 50
    N_REPEAT = 3

    def _timeit(fn):
        best = float("inf")
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    orders = [1, 2, 3, 4]
    baseline_times = []
    propagator_times = []
    batch_times = []

    for order in orders:
        prop = GEqOEPropagator(initial_orbit=orbit, order=order)
        y0 = prop._y0
        bc = prop._bc

        dt_epochs = rng.uniform(60.0, 3600.0, size=N_EPOCHS)
        target_dates = [epoch.shiftedBy(float(dt)) for dt in dt_epochs]
        dt_arr = np.array(dt_epochs)

        def baseline_loop():
            for dt in dt_arr:
                taylor_cart_propagator(
                    tspan=np.array([dt]), y0=y0, p=bc, order=order
                )

        def propagator_loop():
            for date in target_dates:
                prop.propagate(date)

        def batch_fn():
            prop.propagate_array(dt_arr)

        t_baseline = _timeit(baseline_loop)
        t_loop = _timeit(propagator_loop)
        t_batch = _timeit(batch_fn)

        baseline_times.append(t_baseline * 1000)
        propagator_times.append(t_loop * 1000)
        batch_times.append(t_batch * 1000)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(orders))
    width = 0.25

    bars1 = ax.bar(x - width, baseline_times, width, label='Baseline (taylor\_cart per call)', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, propagator_times, width, label='GEqOEPropagator.propagate()', color='#1f77b4', alpha=0.8)
    bars3 = ax.bar(x + width, batch_times, width, label='GEqOEPropagator.propagate\_array()', color='#2ca02c', alpha=0.8)

    ax.set_xlabel('Taylor Order', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'GEqOE Performance Benchmark ({N_EPOCHS} epochs)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Order {o}' for o in orders])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "geqoe_benchmark.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'geqoe_benchmark.png'}")


def main():
    print("=" * 60)
    print("Generating GEqOE benchmark plot")
    print("=" * 60)
    print()
    generate_geqoe_benchmark_plot()
    print()
    print("Plot generated successfully!")


if __name__ == "__main__":
    main()
