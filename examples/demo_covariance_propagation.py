"""Covariance propagation via State Transition Matrix (STM) demo.

Demonstrates :class:`~astrodyn_core.uncertainty.propagator.STMCovariancePropagator`:
propagating a 6×6 orbital state covariance matrix alongside the trajectory
using Orekit's ``setupMatricesComputation`` / ``MatricesHarvester`` API.

The propagated covariance at time t is:

    P(t) = Φ(t, t₀) · P₀ · Φ(t, t₀)ᵀ

where Φ is the State Transition Matrix extracted from the numerical propagator.

Outputs:
  - ``examples/output/covariance_trajectory.yaml`` — sampled states
  - ``examples/output/covariance_series.yaml``     — propagated covariances
  - ``examples/output/covariance_series.h5``       — HDF5 covariance (optional)

Run with:
    conda run -n astrodyn-core-env python examples/demo_covariance_propagation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir

setup_orekit_curdir()

from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    OrbitStateRecord,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    StateFileClient,
    UncertaintySpec,
    register_default_orekit_providers,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("examples/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initial orbital state
INITIAL_STATE = OrbitStateRecord(
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

# Initial covariance (diagonal, Cartesian: 3 position + 3 velocity)
#   Position 1σ: 100 m  → variance 10000 m²
#   Velocity 1σ:  0.1 m/s → variance 0.01 m²/s²
INITIAL_COV = np.diag([
    1e4, 1e4, 1e4,    # position variances (m²)
    1e-2, 1e-2, 1e-2, # velocity variances (m²/s²)
])

# 2-hour propagation, sampled every 10 minutes
EPOCH_SPEC = OutputEpochSpec(
    start_epoch="2026-02-19T00:00:00Z",
    end_epoch="2026-02-19T02:00:00Z",
    step_seconds=600.0,
)

# STM uncertainty configuration
UNCERTAINTY_SPEC = UncertaintySpec(
    method="stm",
    orbit_type="CARTESIAN",   # covariance in Cartesian space
    include_mass=False,        # 6×6 (not 7×7)
    stm_name="stm",
)

# ---------------------------------------------------------------------------
# Build numerical propagator
# ---------------------------------------------------------------------------

ctx = BuildContext.from_state_record(INITIAL_STATE)
registry = ProviderRegistry()
register_default_orekit_providers(registry)
factory = PropagatorFactory(registry=registry)

spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=float(INITIAL_STATE.mass_kg),
    integrator=IntegratorSpec(
        kind="dp853",
        min_step=1.0e-6,
        max_step=300.0,
        position_tolerance=1.0e-3,
    ),
)
builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

print("Numerical propagator built.")
print(f"Initial 1σ position: {np.sqrt(INITIAL_COV[0, 0]):.1f} m")
print(f"Initial 1σ velocity: {np.sqrt(INITIAL_COV[3, 3]):.4f} m/s")

# ---------------------------------------------------------------------------
# Propagate state + covariance
# ---------------------------------------------------------------------------

client = StateFileClient()

print("\n--- Propagating state + covariance ---")
state_series, cov_series = client.propagate_with_covariance(
    propagator,
    INITIAL_COV,
    EPOCH_SPEC,
    spec=UNCERTAINTY_SPEC,
    frame="GCRF",
    series_name="trajectory",
    covariance_name="covariance",
    state_output_path=OUTPUT_DIR / "covariance_trajectory.yaml",
    covariance_output_path=OUTPUT_DIR / "covariance_series.yaml",
)

# Also save HDF5
try:
    client.save_covariance_series(OUTPUT_DIR / "covariance_series.h5", cov_series)
    print(f"HDF5 covariance saved: {OUTPUT_DIR / 'covariance_series.h5'}")
except ImportError:
    print("[Note] h5py not available — HDF5 output skipped.")

# ---------------------------------------------------------------------------
# Print covariance evolution summary
# ---------------------------------------------------------------------------

print(f"\n{'Epoch':>30}  {'1σ-pos (m)':>12}  {'1σ-vel (m/s)':>14}  {'trace':>14}")
print("-" * 76)
for rec in cov_series.records:
    cov = rec.to_numpy()
    pos_sigma = np.sqrt(np.trace(cov[:3, :3]) / 3.0)
    vel_sigma = np.sqrt(np.trace(cov[3:, 3:]) / 3.0)
    trace = np.trace(cov)
    print(f"  {rec.epoch:>30}  {pos_sigma:>12.2f}  {vel_sigma:>14.6f}  {trace:>14.4e}")

# ---------------------------------------------------------------------------
# Verify PSD property
# ---------------------------------------------------------------------------

print("\n--- PSD verification ---")
all_psd = True
for rec in cov_series.records:
    cov = rec.to_numpy()
    eigenvalues = np.linalg.eigvalsh(cov)
    if np.any(eigenvalues < -1e-6):
        print(f"  WARNING: Non-PSD at {rec.epoch}: min eigenvalue={eigenvalues.min():.3e}")
        all_psd = False

if all_psd:
    print("  All propagated covariances are positive semi-definite. ✓")

# ---------------------------------------------------------------------------
# Load back and verify round-trip
# ---------------------------------------------------------------------------

loaded = client.load_covariance_series(OUTPUT_DIR / "covariance_series.yaml")
np.testing.assert_array_almost_equal(
    cov_series.matrices_numpy(),
    loaded.matrices_numpy(),
    decimal=8,
)
print("\nYAML round-trip verification passed. ✓")

print(f"\nOutputs:")
print(f"  {OUTPUT_DIR / 'covariance_trajectory.yaml'}")
print(f"  {OUTPUT_DIR / 'covariance_series.yaml'}")
print("Done.")
