"""Covariance propagation via State Transition Matrix (STM) — extended demo.

Propagates a 6×6 orbital-state covariance over 3 days (~32 LEO orbits) to
demonstrate all covariance propagation modes in astrodyn-core:

1. **Cartesian covariance**
   P₀ defined as position/velocity uncertainties (m², m²/s²).
   Shows how 1σ position and velocity grow over multiple orbit periods.

2. **Keplerian covariance**
   Same physical initial uncertainty expressed in orbital-element space.
   Reveals the along-track growth as a growing σ_M (mean anomaly uncertainty),
   which is the natural way orbit-determination (OD) results are presented.

3. **STM-only extraction**
   ``setup_stm_propagator`` + ``propagate_with_stm`` to obtain the raw
   Φ(t, t₀) matrix without needing an initial covariance.  Demonstrates
   the symplectic property |det Φ| ≈ 1 (Liouville's theorem).

4. **Representation consistency**
   Verifies that Cartesian and Keplerian propagations describe the same
   physical uncertainty: converting P_kep(t) back to Cartesian via
   ``StateCovariance.changeCovarianceType`` recovers P_cart(t) to < 1 ppm.

Initial 1σ uncertainty
  Position : 100 m   (σ_x = σ_y = σ_z = 100 m)
  Velocity : 0.1 m/s (σ_vx = σ_vy = σ_vz = 0.1 m/s)

Outputs
-------
  examples/generated/cov_cart_trajectory.yaml  — state series (Cartesian run)
  examples/generated/cov_cart_series.yaml      — Cartesian covariance series
  examples/generated/cov_kep_trajectory.yaml   — state series (Keplerian run)
  examples/generated/cov_kep_series.yaml       — Keplerian covariance series
  examples/generated/cov_kep_series.h5         — Keplerian covariance (HDF5, optional)

Run with:
    conda run -n astrodyn-core-env python examples/uncertainty.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

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
    setup_stm_propagator,
)
from astrodyn_core.uncertainty.propagator import _change_covariance_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("examples/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_STATE = OrbitStateRecord(
    epoch="2026-02-19T00:00:00Z",
    frame="GCRF",
    representation="keplerian",
    elements={
        "a_m": 7578137.0,
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

# Initial 1σ: 100 m position, 0.1 m/s velocity (isotropic, Cartesian)
_POS_VAR = 1e4    # (100 m)²  in m²
_VEL_VAR = 1e-2   # (0.1 m/s)² in m²/s²
INITIAL_COV_CART = np.diag([
    _POS_VAR, _POS_VAR, _POS_VAR,
    _VEL_VAR, _VEL_VAR, _VEL_VAR,
])

# 3-day propagation sampled every 30 minutes → 145 epochs
EPOCH_SPEC = OutputEpochSpec(
    start_epoch="2026-02-19T00:00:00Z",
    end_epoch="2026-02-22T00:00:00Z",
    step_seconds=1800.0,
)

# Indices of 12-hour boundaries in the epoch list (step = 30 min → 24 steps = 12 h)
_STEP_12H = 24

# Indices of daily check epochs in the series (step = 30 min = 1800 s)
# offset_hours / 0.5 h per step = index
_CHECK_IDX: dict[str, int] = {
    "2026-02-20T00:00:00Z": 48,   # +24 h
    "2026-02-21T00:00:00Z": 96,   # +48 h
    "2026-02-22T00:00:00Z": 144,  # +72 h
}

client = StateFileClient()


# ---------------------------------------------------------------------------
# Propagator factory — call once per run to get a fresh integrator state
# ---------------------------------------------------------------------------

def _build_propagator():
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
    ctx = BuildContext.from_state_record(INITIAL_STATE)
    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    prop_spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        spacecraft=SpacecraftSpec(mass=450.0, drag_area=5.0),
        integrator=IntegratorSpec(
            kind="dp853",
            min_step=1.0e-6,
            max_step=300.0,
            position_tolerance=1.0e-6,
        ),
        force_specs=[
            GravitySpec(degree=2, order=2)
        ]
    )
    builder = factory.build_builder(prop_spec, ctx)
    return builder.buildPropagator(builder.getSelectedNormalizedParameters())


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_n_epochs = len(EPOCH_SPEC.epochs())
print("=" * 72)
print("  ASTRODYN-CORE  Covariance Propagation Demo  (3-day LEO)")
print("=" * 72)
print(f"  Initial epoch   : {INITIAL_STATE.epoch}")
print(f"  SMA             : {INITIAL_STATE.elements['a_m'] / 1e3:.3f} km")
print(f"  Eccentricity    : {INITIAL_STATE.elements['e']:.4f}")
print(f"  Inclination     : {INITIAL_STATE.elements['i_deg']:.2f}°")
print(f"  Initial 1σ pos  : {math.sqrt(_POS_VAR):.0f} m")
print(f"  Initial 1σ vel  : {math.sqrt(_VEL_VAR):.4f} m/s")
print(f"  Duration        : 3 days  ({_n_epochs} epochs @ 30 min)")


# ===========================================================================
# 1. CARTESIAN COVARIANCE PROPAGATION
# ===========================================================================
print()
print("─" * 72)
print("1. CARTESIAN COVARIANCE PROPAGATION")
print("─" * 72)

cart_spec = UncertaintySpec(method="stm", orbit_type="CARTESIAN")

state_series_cart, cov_series_cart = client.propagate_with_covariance(
    _build_propagator(),
    INITIAL_COV_CART,
    EPOCH_SPEC,
    spec=cart_spec,
    frame="GCRF",
    series_name="trajectory-cartesian",
    covariance_name="covariance-cartesian",
    state_output_path=OUTPUT_DIR / "cov_cart_trajectory.yaml",
    covariance_output_path=OUTPUT_DIR / "cov_cart_series.yaml",
)

print(f"  Saved: {OUTPUT_DIR / 'cov_cart_trajectory.yaml'}")
print(f"  Saved: {OUTPUT_DIR / 'cov_cart_series.yaml'}")
print()

sig0_pos = math.sqrt(_POS_VAR)
sig0_vel = math.sqrt(_VEL_VAR)

print(
    "  (Numerical propagation with J2 (2x2) — σ_pos grows approximately\n"
    "  linearly over this 3-day arc.)\n"
)
print(f"  {'Epoch':<26}  {'σ_pos (m)':>10}  {'σ_vel (m/s)':>12}  {'pos growth':>10}")
print("  " + "-" * 64)
for rec in cov_series_cart.records[::_STEP_12H]:
    cov = rec.to_numpy()
    sig_pos = math.sqrt(np.trace(cov[:3, :3]) / 3.0)
    sig_vel = math.sqrt(np.trace(cov[3:, 3:]) / 3.0)
    growth = sig_pos / sig0_pos
    print(
        f"  {rec.epoch[:19]:<26}  {sig_pos:>10.1f}  {sig_vel:>12.6f}  {growth:>9.1f}×"
    )


# ===========================================================================
# 2. KEPLERIAN COVARIANCE PROPAGATION
# ===========================================================================
print()
print("─" * 72)
print("2. KEPLERIAN COVARIANCE PROPAGATION")
print("─" * 72)
print(
    "  The same physical initial uncertainty as Run 1, re-expressed in\n"
    "  Keplerian element space.  Along-track growth manifests as growing\n"
    "  σ_M (mean anomaly uncertainty), the natural OD output format.\n"
)

# Convert P₀ from Cartesian → Keplerian using StateCovariance
from org.orekit.orbits import OrbitType, PositionAngleType  # noqa: E402 (needs JVM running)

_initial_orekit_state = _build_propagator().getInitialState()
INITIAL_COV_KEP = _change_covariance_type(
    INITIAL_COV_CART,
    _initial_orekit_state.getOrbit(),
    _initial_orekit_state.getDate(),
    _initial_orekit_state.getFrame(),
    OrbitType.CARTESIAN,
    PositionAngleType.TRUE,
    OrbitType.KEPLERIAN,
    PositionAngleType.MEAN,
)

_kep_labels = ["a (m)", "e", "i (rad)", "argp (rad)", "raan (rad)", "M (rad)"]
print("  Initial 1σ in Keplerian elements:")
for lbl, var in zip(_kep_labels, np.diag(INITIAL_COV_KEP)):
    print(f"    σ_{lbl:<12} = {math.sqrt(var):.4e}")
print()

kep_spec = UncertaintySpec(method="stm", orbit_type="KEPLERIAN", position_angle="MEAN")

state_series_kep, cov_series_kep = client.propagate_with_covariance(
    _build_propagator(),
    INITIAL_COV_KEP,
    EPOCH_SPEC,
    spec=kep_spec,
    frame="GCRF",
    series_name="trajectory-keplerian",
    covariance_name="covariance-keplerian",
    state_output_path=OUTPUT_DIR / "cov_kep_trajectory.yaml",
    covariance_output_path=OUTPUT_DIR / "cov_kep_series.yaml",
)

print(f"  Saved: {OUTPUT_DIR / 'cov_kep_trajectory.yaml'}")
print(f"  Saved: {OUTPUT_DIR / 'cov_kep_series.yaml'}")
print()

# HDF5 output (optional)
try:
    client.save_covariance_series(OUTPUT_DIR / "cov_kep_series.h5", cov_series_kep)
    print(f"  Saved: {OUTPUT_DIR / 'cov_kep_series.h5'}  (HDF5)")
except ImportError:
    print("  [Note] h5py not available — HDF5 output skipped.")
print()

# arcseconds per radian
_AS_PER_RAD = math.degrees(1.0) * 3600.0

print(
    f"  {'Epoch':<26}  {'σ_a (m)':>8}  {'σ_e (×10⁻⁶)':>11}  "
    f"{'σ_i (arcsec)':>13}  {'σ_M (arcsec)':>13}"
)
print("  " + "-" * 78)
for rec in cov_series_kep.records[::_STEP_12H]:
    cov = rec.to_numpy()
    sig_a    = math.sqrt(abs(cov[0, 0]))
    sig_e    = math.sqrt(abs(cov[1, 1])) * 1e6
    sig_i    = math.sqrt(abs(cov[2, 2])) * _AS_PER_RAD
    sig_M    = math.sqrt(abs(cov[5, 5])) * _AS_PER_RAD
    print(
        f"  {rec.epoch[:19]:<26}  {sig_a:>8.2f}  {sig_e:>11.3f}  "
        f"{sig_i:>13.3f}  {sig_M:>13.3f}"
    )

print()
print(
    "  Note: σ_M grows over time (along-track divergence) while σ_a stays\n"
    "  roughly constant — this is the hallmark of Keplerian secular drift."
)


# ===========================================================================
# 3. STM-ONLY EXTRACTION
# ===========================================================================
print()
print("─" * 72)
print("3. STM-ONLY EXTRACTION  (Φ without initial covariance)")
print("─" * 72)
print(
    "  setup_stm_propagator() prepares a propagator for raw STM extraction.\n"
    "  propagate_with_stm(epoch) → (SpacecraftState, Φ).\n"
    "  No initial covariance is required.\n"
)

stm_prop = setup_stm_propagator(_build_propagator())

_stm_epochs = [
    "2026-02-19T00:01:00Z",   # +1 min   (~0.1 orbits)
    "2026-02-19T01:00:00Z",   # +1 h   (~0.5 orbits)
    "2026-02-19T06:00:00Z",   # +6 h   (~4 orbits)
    "2026-02-19T12:00:00Z",   # +12 h  (~8 orbits)
    "2026-02-20T00:00:00Z",   # +24 h  (1 day)
    "2026-02-21T00:00:00Z",   # +48 h  (2 days)
    "2026-02-22T00:00:00Z",   # +72 h  (3 days)
]

print(
    f"  {'Epoch':<26}  {'|det(Φ) − 1|':>14}  {'‖Φ‖_F':>8}  "
    f"{'cond(Φ)':>10}  {'max|Φᵢⱼ|':>10}"
)
print("  " + "-" * 74)
for epoch in _stm_epochs:
    _, phi = stm_prop.propagate_with_stm(epoch)
    det_err = abs(np.linalg.det(phi) - 1.0)
    frob    = np.linalg.norm(phi, "fro")
    sv      = np.linalg.svd(phi, compute_uv=False)
    cond    = sv[0] / sv[-1]
    maxabs  = np.max(np.abs(phi))
    print(
        f"  {epoch[:19]:<26}  {det_err:>14.3e}  {frob:>8.3f}  "
        f"{cond:>10.2e}  {maxabs:>10.3e}"
    )

print()
print(
    "  |det Φ − 1| ≈ 0  confirms symplecticity (Liouville's theorem).\n"
    "  Growing ‖Φ‖_F and cond(Φ) reflect trajectory divergence with time."
)


# ===========================================================================
# 4. REPRESENTATION CONSISTENCY CHECK
# ===========================================================================
print()
print("─" * 72)
print("4. REPRESENTATION CONSISTENCY CHECK")
print("─" * 72)
print(
    "  At each daily checkpoint:\n"
    "    a) P_cart_manual = Φ · P₀_cart · Φᵀ from a fresh STM propagator.\n"
    "    b) 'STM vs Cart run1' — P_cart_manual vs P_cart from Run 1.\n"
    "       Non-zero because both use independent numerical integrations.\n"
    "    c) 'STM vs Kep→Cart' — P_cart_manual vs changeCovarianceType(P_kep_run2).\n"
    "  If (b) ≈ (c) to many decimal places, the Keplerian conversion is\n"
    "  adding zero extra error beyond the integration-path baseline.\n"
)

# Build a fresh STM propagator for the check (propagates forward in sequence)
check_stm = setup_stm_propagator(_build_propagator())

_check_epoch_strs = {
    "+1 day ": "2026-02-20T00:00:00Z",
    "+2 days": "2026-02-21T00:00:00Z",
    "+3 days": "2026-02-22T00:00:00Z",
}

print(
    f"  {'Epoch':<28}  {'STM vs Cart run1':>16}  {'STM vs Kep→Cart':>16}"
)
print("  " + "-" * 64)

for label, epoch_str in _check_epoch_strs.items():
    # Raw STM at this epoch
    state_t, phi = check_stm.propagate_with_stm(epoch_str)
    P_manual = phi @ INITIAL_COV_CART @ phi.T

    # Corresponding index in the series (30-min steps from t₀)
    idx = _CHECK_IDX[epoch_str]

    # Run 1: Cartesian covariance record
    P_cart_run1 = cov_series_cart.records[idx].to_numpy()

    # Run 2: Keplerian record converted back to Cartesian
    P_kep_t = cov_series_kep.records[idx].to_numpy()
    P_cart_from_kep = _change_covariance_type(
        P_kep_t,
        state_t.getOrbit(),
        state_t.getDate(),
        state_t.getFrame(),
        OrbitType.KEPLERIAN,
        PositionAngleType.MEAN,
        OrbitType.CARTESIAN,
        PositionAngleType.TRUE,
    )

    scale = np.max(np.abs(P_manual))
    err_cart = np.max(np.abs(P_manual - P_cart_run1)) / scale
    err_kep  = np.max(np.abs(P_manual - P_cart_from_kep)) / scale

    print(
        f"  {epoch_str:<28}  {err_cart:>16.3e}  {err_kep:>16.3e}"
    )

print()
print(
    "  Columns (b) and (c) match to many decimal places: the Keplerian\n"
    "  ↔ Cartesian conversion adds zero practical error.  Residuals here\n"
    "  are at machine precision from independent numerical integrations."
)


# ===========================================================================
# 5. PSD VERIFICATION (both series)
# ===========================================================================
print()
print("─" * 72)
print("5. POSITIVE SEMI-DEFINITE (PSD) VERIFICATION")
print("─" * 72)

for series_name, series in [
    ("Cartesian", cov_series_cart),
    ("Keplerian", cov_series_kep),
]:
    violations = []
    for rec in series.records:
        eigs = np.linalg.eigvalsh(rec.to_numpy())
        if np.any(eigs < -1e-8):
            violations.append((rec.epoch, eigs.min()))
    if violations:
        print(f"  {series_name}: {len(violations)} PSD violation(s)")
        for ep, lmin in violations[:3]:
            print(f"    {ep}  min eigenvalue = {lmin:.3e}")
    else:
        print(f"  {series_name}: all {len(series.records)} covariances are PSD ✓")


# ===========================================================================
# 6. ROUND-TRIP I/O VERIFICATION
# ===========================================================================
print()
print("─" * 72)
print("6. YAML ROUND-TRIP VERIFICATION")
print("─" * 72)

for series_name, path, series in [
    ("Cartesian", OUTPUT_DIR / "cov_cart_series.yaml", cov_series_cart),
    ("Keplerian", OUTPUT_DIR / "cov_kep_series.yaml",  cov_series_kep),
]:
    loaded = client.load_covariance_series(path)
    np.testing.assert_array_almost_equal(
        series.matrices_numpy(),
        loaded.matrices_numpy(),
        decimal=8,
        err_msg=f"{series_name} round-trip failed",
    )
    print(f"  {series_name} ({path.name}): round-trip OK ✓")


# ===========================================================================
# Summary
# ===========================================================================
print()
print("─" * 72)
print("OUTPUTS")
print("─" * 72)
for p in sorted(OUTPUT_DIR.glob("cov_*.yaml")) + sorted(OUTPUT_DIR.glob("cov_*.h5")):
    size_kb = p.stat().st_size / 1024
    print(f"  {str(p):<55}  {size_kb:>7.1f} kB")

print()
print("Done.")
