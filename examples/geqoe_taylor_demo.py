"""GEqOE Taylor Propagator Demo — interactive playground.

Run: conda run -n astrodyn-core-env python examples/geqoe_taylor_demo.py

Demonstrates:
  1. Cartesian <-> GEqOE conversions
  2. 12-day J2 propagation with accuracy check
  3. Step-by-step propagation with step size history
  4. Dense output via propagate_grid
  5. STM (State Transition Matrix) computation
  6. Ground truth comparison (GEqOE vs Cowell vs paper Dromo)
  7. Performance summary
"""

import time
import numpy as np

from astrodyn_core.geqoe_taylor.constants import MU, RE
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.integrator import (
    build_state_integrator,
    build_stm_integrator,
    propagate,
    propagate_grid,
    extract_stm,
)
from astrodyn_core.geqoe_taylor.utils import K_to_L

np.set_printoptions(precision=10, linewidth=120)

pert = J2Perturbation()

# ── Initial state: LEO circular i=45° (paper case a, Table 2) ──────────────
r0 = np.array([7178.1366, 0.0, 0.0])                          # km
v0 = np.array([0.0, 5.269240572916780, 5.269240572916780])    # km/s


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. CARTESIAN <-> GEqOE CONVERSIONS")
print("=" * 70)

ic = cart2geqoe(r0, v0, MU, pert)
labels = ["nu (rad/s)", "p1", "p2", "K (rad)", "q1", "q2"]
print("\nCartesian -> GEqOE:")
for lbl, val in zip(labels, ic):
    print(f"  {lbl:12s} = {val: .12e}")

r_back, v_back = geqoe2cart(ic, MU, pert)
print(f"\nRound-trip errors:")
print(f"  position: {np.linalg.norm(r_back - r0):.2e} km")
print(f"  velocity: {np.linalg.norm(v_back - v0):.2e} km/s")

L0 = K_to_L(ic[3], ic[1], ic[2])
print(f"\n  L (mean longitude) from K: {L0:.12e} rad")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 12-DAY PROPAGATION vs PAPER REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. 12-DAY J2 PROPAGATION")
print("=" * 70)

t_build = time.perf_counter()
ta, par_map = build_state_integrator(pert, ic, tol=1e-15)
t_build = time.perf_counter() - t_build

print(f"\nIntegrator built in {t_build*1000:.0f} ms")
print(f"  Taylor order: {ta.order}")
print(f"  Parameters: {par_map}")

t_final = 12.0 * 86400.0  # 12 days in seconds
t_prop = time.perf_counter()
outcome = ta.propagate_until(t_final)
t_prop = time.perf_counter() - t_prop

rf, vf = geqoe2cart(ta.state, MU, pert)

r_ref = np.array([-5398.929377366906, -390.257240638229, -4693.719111636971])
v_ref = np.array([2.214482567493, -6.845637008953, -1.977748618717])

print(f"\nPropagated {t_final/86400:.0f} days in {t_prop*1000:.1f} ms  ({outcome[3]} steps)")
print(f"\nFinal Cartesian state:")
print(f"  r = {rf} km")
print(f"  v = {vf} km/s")
print(f"\nPaper reference (Appendix C):")
print(f"  r = {r_ref} km")
print(f"  v = {v_ref} km/s")
print(f"\nErrors:")
print(f"  position: {np.linalg.norm(rf - r_ref):.2e} km  ({np.linalg.norm(rf - r_ref)*1e3:.4f} m)")
print(f"  velocity: {np.linalg.norm(vf - v_ref):.2e} km/s")
print(f"\nnu conserved: {ta.state[0] == ic[0]}  (delta = {abs(ta.state[0] - ic[0]):.1e})")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STEP-BY-STEP PROPAGATION WITH STEP SIZE HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. STEP-BY-STEP PROPAGATION (1 day)")
print("=" * 70)

ta2, _ = build_state_integrator(pert, ic, tol=1e-15)
times, states = propagate(ta2, 86400.0)

h_steps = np.diff(times)
T_orb = 2 * np.pi / ic[0]

print(f"\n  Steps taken:     {len(h_steps)}")
print(f"  Orbital period:  {T_orb:.1f} s  ({T_orb/60:.1f} min)")
print(f"  Steps per orbit: {len(h_steps) / (86400.0 / T_orb):.1f}")
print(f"  Step sizes (s):  min={h_steps.min():.1f}  max={h_steps.max():.1f}  mean={h_steps.mean():.1f}")

# Show element evolution over 1 day
states_arr = np.array(states)
print(f"\n  Element ranges over 1 day:")
for i, lbl in enumerate(labels):
    lo, hi = states_arr[:, i].min(), states_arr[:, i].max()
    print(f"    {lbl:12s}: [{lo: .8e}, {hi: .8e}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DENSE OUTPUT (propagate_grid)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. DENSE OUTPUT — 1-MINUTE GRID OVER 1 ORBIT")
print("=" * 70)

ta3, _ = build_state_integrator(pert, ic, tol=1e-15)
dt_grid = 60.0  # 1-minute spacing
t_grid = np.arange(0.0, T_orb + dt_grid, dt_grid)
grid_states = propagate_grid(ta3, t_grid)

print(f"\n  Grid points: {len(t_grid)} (dt = {dt_grid:.0f} s)")
print(f"  Output shape: {grid_states.shape}")

# Convert all grid states to Cartesian
positions = np.zeros((len(t_grid), 3))
for i, s in enumerate(grid_states):
    positions[i], _ = geqoe2cart(s, MU, pert)

r_mag = np.linalg.norm(positions, axis=1)
z_comp = positions[:, 2]

print(f"\n  Altitude range: {r_mag.min() - RE:.1f} to {r_mag.max() - RE:.1f} km")
print(f"  z range:        {z_comp.min():.1f} to {z_comp.max():.1f} km")
print(f"\n  First 5 grid states (GEqOE):")
for i in range(5):
    t_min = t_grid[i] / 60
    print(f"    t={t_min:5.1f} min: nu={grid_states[i,0]:.8e}  K={grid_states[i,3]:+.6f} rad")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STATE TRANSITION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. STATE TRANSITION MATRIX (1 orbit)")
print("=" * 70)

t_build = time.perf_counter()
ta_v, _ = build_stm_integrator(pert, ic, tol=1e-15)
t_build = time.perf_counter() - t_build

t_prop = time.perf_counter()
ta_v.propagate_until(T_orb)
t_prop = time.perf_counter() - t_prop

y_final, phi = extract_stm(ta_v.state)
print(f"\n  Build time:  {t_build*1000:.0f} ms  (42 DOF variational system)")
print(f"  Prop time:   {t_prop*1000:.1f} ms")

print(f"\n  Final state after 1 orbit:")
for lbl, val in zip(labels, y_final):
    print(f"    {lbl:12s} = {val: .12e}")

print(f"\n  STM (6x6):")
np.set_printoptions(precision=4, linewidth=120, suppress=True)
print(phi)
np.set_printoptions(precision=10, linewidth=120, suppress=False)

print(f"\n  STM determinant: {np.linalg.det(phi):.10f}  (should be ~1.0 for symplectic)")
print(f"  STM max element: {np.abs(phi).max():.2f}")
print(f"  STM diagonal:    {np.diag(phi)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GROUND TRUTH COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. GROUND TRUTH: GEqOE vs COWELL vs PAPER DROMO (12 days)")
print("=" * 70)

from astrodyn_core.geqoe_taylor.cowell import propagate_cowell, propagate_cowell_heyoka

# Heyoka Cowell (highest accuracy Cartesian reference)
t0 = time.perf_counter()
r_cow_hy, v_cow_hy = propagate_cowell_heyoka(r0, v0, t_final)
t_cow_hy = time.perf_counter() - t0

# Scipy DOP853 Cowell
t0 = time.perf_counter()
r_cow_sp, v_cow_sp = propagate_cowell(r0, v0, t_final)
t_cow_sp = time.perf_counter() - t0

# GEqOE Taylor (reuse rf, vf from section 2)
# Paper Dromo reference
r_ref = np.array([-5398.929377366906, -390.257240638229, -4693.719111636971])
v_ref = np.array([2.214482567493, -6.845637008953, -1.977748618717])

print(f"\n  Ground truth: heyoka Cowell Taylor (tol=1e-15, {t_cow_hy*1000:.0f} ms)")
print(f"    r = {r_cow_hy}")
print(f"    v = {v_cow_hy}")

print(f"\n  {'Method':<25s} {'pos (km)':>12s} {'vel (km/s)':>12s} {'pos (m)':>10s}")
print("  " + "-" * 62)
for name, r, v in [
    ("GEqOE Taylor (heyoka)", rf, vf),
    ("Paper Dromo (App. C)", r_ref, v_ref),
    ("Scipy DOP853", r_cow_sp, v_cow_sp),
]:
    dr = np.linalg.norm(r - r_cow_hy)
    dv = np.linalg.norm(v - v_cow_hy)
    print(f"  {name:<25s} {dr:12.3e} {dv:12.3e} {dr*1e3:10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. PERFORMANCE SUMMARY")
print("=" * 70)

# Fresh build + propagate for clean timing
ta_fresh, _ = build_state_integrator(pert, ic, tol=1e-15)
t0 = time.perf_counter()
oc = ta_fresh.propagate_until(365.25 * 86400.0)  # 1 year
t1 = time.perf_counter()

print(f"\n  1-year propagation:")
print(f"    Steps:     {oc[3]}")
print(f"    Wall time: {(t1-t0)*1000:.1f} ms")
print(f"    Step size: {oc[1]:.1f} - {oc[2]:.1f} s")

rf_year, vf_year = geqoe2cart(ta_fresh.state, MU, pert)
print(f"    Final r:   {rf_year} km")
print(f"    Final |r|: {np.linalg.norm(rf_year):.4f} km  (alt = {np.linalg.norm(rf_year)-RE:.1f} km)")

print("\n" + "=" * 70)
print("Done! Edit this script to experiment with different orbits,")
print("tolerances, and propagation durations.")
print("=" * 70)
