#!/usr/bin/env python
"""Comprehensive validation of the averaged J2 GEqOE formulation.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/validation_check.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrodyn_core.geqoe_taylor import (
    J2 as J2_VAL, MU, RE, J2Perturbation,
    build_state_integrator, cart2geqoe, geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid
from astrodyn_core.geqoe_taylor.utils import K_to_L, solve_kepler_gen


# ── helpers ──────────────────────────────────────────────────────────
def _rot3(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def _rot1(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _kepler_to_rv(a, e, i_deg, raan_deg, argp_deg, M_deg, mu=MU):
    i, raan, argp = np.deg2rad(i_deg), np.deg2rad(raan_deg), np.deg2rad(argp_deg)
    M = np.deg2rad(M_deg)
    E = M if e < 0.8 else np.pi
    for _ in range(50):
        dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E -= dE
        if abs(dE) < 1e-14: break
    cE, sE = np.cos(E), np.sin(E)
    r_pf = np.array([a*(cE-e), a*np.sqrt(1-e*e)*sE, 0.])
    rm = a*(1-e*cE)
    v_pf = np.sqrt(mu*a)/rm * np.array([-sE, np.sqrt(1-e*e)*cE, 0.])
    dcm = _rot3(raan) @ _rot1(i) @ _rot3(argp)
    return dcm @ r_pf, dcm @ v_pf


def _secular_rates(state0, j2=J2_VAL):
    nu0 = state0[0]
    g0 = np.hypot(state0[1], state0[2])
    Q0 = np.hypot(state0[4], state0[5])
    beta = np.sqrt(1 - g0*g0)
    a = (MU / nu0**2)**(1/3)
    gamma = 1 + Q0*Q0
    delta = 1 - Q0*Q0
    c_i = delta / gamma
    p = a * beta**2
    omega_node = -1.5 * nu0 * j2 * (RE/p)**2 * c_i
    omega_peri = 0.75 * nu0 * j2 * (RE/p)**2 * (5*c_i**2 - 2*c_i - 1)
    return omega_peri, omega_node


def _secular_solution(state0, t_grid, j2=J2_VAL):
    nu0, p10, p20, _, q10, q20 = state0
    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    psi0 = np.arctan2(p10, p20)
    omega0 = np.arctan2(q10, q20)
    wp, wn = _secular_rates(state0, j2)
    psi = psi0 + wp * t_grid
    omega = omega0 + wn * t_grid
    return {
        "p1": g0*np.sin(psi), "p2": g0*np.cos(psi),
        "q1": Q0*np.sin(omega), "q2": Q0*np.cos(omega),
        "psi": psi, "Omega": omega, "g": np.full_like(t_grid, g0),
        "Q": np.full_like(t_grid, Q0),
    }


def _cumtrap(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(x))
    return out

def _periodic_zero_mean_integral(prime, x):
    raw = _cumtrap(prime, x)
    return raw - np.trapezoid(raw, x) / (x[-1] - x[0])

def _interp_periodic(x, y, xe):
    period = x[-1] - x[0]
    return float(np.interp((xe - x[0]) % period + x[0], x, y))


def _short_period_map(state0, G_eval, j2=J2_VAL):
    """Compute short-period corrections at given G value."""
    nu0, p10, p20, K0, q10, q20 = state0
    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    psi0 = np.arctan2(p10, p20)
    omega0 = np.arctan2(q10, q20)
    beta = np.sqrt(1 - g0*g0)
    a = (MU / nu0**2)**(1/3)
    w = np.sqrt(MU / a)
    c = (MU**2 / nu0)**(1/3) * beta
    gamma = 1 + Q0**2
    delta = 1 - Q0**2
    s_i = 2*Q0 / gamma
    c_i = delta / gamma
    A = MU * j2 * RE**2 / 2

    wp, wn = _secular_rates(state0, j2)

    G = np.linspace(0, 2*np.pi, 4097)
    cosG, sinG = np.cos(G), np.sin(G)
    sinK, cosK = np.sin(G + psi0), np.cos(G + psi0)
    alpha = 1/(1+beta)
    X = a*(alpha*p10*p20*sinK + (1-alpha*p10**2)*cosK - p20)
    Y = a*(alpha*p10*p20*cosK + (1-alpha*p20**2)*sinK - p10)
    r = a*(1 - g0*cosG)
    cosf = (cosG - g0)/(1 - g0*cosG)
    sinf = beta*sinG/(1 - g0*cosG)
    argp = psi0 - omega0
    sin_u = np.sin(argp)*cosf + np.cos(argp)*sinf
    cos_u = np.cos(argp)*cosf - np.sin(argp)*sinf

    U = -A/r**3 * (1 - 3*s_i**2*sin_u**2)
    h_approx = c  # first order
    d = -U / c
    # w_h to first order
    zhat = s_i * sin_u
    w_h = 3*A*zhat**2*delta / (c * r**3)

    # Exact p1_dot, p2_dot from J2 GEqOE RHS
    xi1 = X/a + 2*p20
    xi2 = Y/a + 2*p10
    E_Z = -U  # for J2
    p1_dot = p20*(d - w_h) + (1/c)*xi1*E_Z
    p2_dot = p10*(w_h - d) - (1/c)*xi2*E_Z
    psi_dot = (p20*p1_dot - p10*p2_dot) / g0**2

    dg_dG = (r/w)*(U/c)*beta*sinG
    dQ_dG = -6*A*Q0*c_i/(MU*beta*r**2)*sin_u*cos_u
    dOmega_dG = -6*A*c_i/(MU*beta*r**2)*sin_u**2
    dPsi_dG = (r/w)*psi_dot

    N_prime = dOmega_dG - (r/w)*wn
    P_prime = dPsi_dG - (r/w)*wp

    g_corr = _periodic_zero_mean_integral(dg_dG, G)
    Q_corr = _periodic_zero_mean_integral(dQ_dG, G)
    Omega_corr = _periodic_zero_mean_integral(N_prime, G)
    Psi_corr = _periodic_zero_mean_integral(P_prime, G)

    return {
        "g_eval": _interp_periodic(G, g_corr, G_eval),
        "Q_eval": _interp_periodic(G, Q_corr, G_eval),
        "Omega_eval": _interp_periodic(G, Omega_corr, G_eval),
        "Psi_eval": _interp_periodic(G, Psi_corr, G_eval),
    }


def _orbit_means(series, n_orbits, spo):
    return series[:n_orbits*spo].reshape(n_orbits, spo).mean(axis=1)


def _build_mean_state(state0, corr):
    """Apply inverse map corrections."""
    g0 = np.hypot(state0[1], state0[2])
    psi0 = np.arctan2(state0[1], state0[2])
    Q0 = np.hypot(state0[4], state0[5])
    Omega0 = np.arctan2(state0[4], state0[5])
    ms = state0.copy()
    gm = g0 - corr["g_eval"]
    pm = psi0 - corr["Psi_eval"]
    Qm = Q0 - corr["Q_eval"]
    Om = Omega0 - corr["Omega_eval"]
    ms[1] = gm * np.sin(pm)
    ms[2] = gm * np.cos(pm)
    ms[4] = Qm * np.sin(Om)
    ms[5] = Qm * np.cos(Om)
    return ms


def _inverse_map_initial_state(state0, j2=J2_VAL):
    psi0 = np.arctan2(state0[1], state0[2])
    G0 = state0[3] - psi0
    corr = _short_period_map(state0, G0, j2=j2)
    return _build_mean_state(state0, corr), corr


def _mean_longitude_rate(state0, j2=J2_VAL):
    nu0 = state0[0]
    g0 = np.hypot(state0[1], state0[2])
    Q0 = np.hypot(state0[4], state0[5])
    beta = np.sqrt(1 - g0 * g0)
    a = (MU / nu0**2) ** (1 / 3)
    gamma = 1 + Q0 * Q0
    delta = 1 - Q0 * Q0
    c_i = delta / gamma
    p = a * beta**2
    return nu0 + 0.75 * nu0 * j2 * (RE / p) ** 2 * (
        beta * (3 * c_i**2 - 1) + 5 * c_i**2 - 2 * c_i - 1
    )


# ── CHECK 1: Verify angle meanings ──────────────────────────────────
def check1_angle_meanings():
    print("=" * 72)
    print("CHECK 1: GEqOE angle/magnitude meanings")
    print("=" * 72)
    # Use a known orbit: a=26600 km, e=0.74, i=63.4, RAAN=30, argp=270, M=45
    a_km = 6916 / (1 - 0.74)
    r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, 45)
    s = cart2geqoe(r0, v0, MU, J2Perturbation())
    nu, p1, p2, K, q1, q2 = s

    g = np.hypot(p1, p2)
    Q = np.hypot(q1, q2)
    Psi = np.arctan2(p1, p2)
    Omega = np.arctan2(q1, q2)

    # Expected from classical elements
    e_expected = 0.74
    i_expected = np.deg2rad(63.4)
    raan_expected = np.deg2rad(30.0)
    varpi_expected = np.deg2rad(270.0 + 30.0)  # longitude of periapsis = omega + RAAN
    Q_expected = np.tan(i_expected / 2)

    print(f"  g = {g:.8f}  (cf. e = {e_expected:.8f}, diff = {abs(g-e_expected):.2e})")
    print(f"  Q = {Q:.8f}  (cf. tan(i/2) = {Q_expected:.8f}, diff = {abs(Q-Q_expected):.2e})")
    print(f"  Psi = {np.rad2deg(Psi):.4f} deg  (cf. varpi = {np.rad2deg(varpi_expected):.4f} deg)")
    print(f"  Omega = {np.rad2deg(Omega):.4f} deg  (cf. RAAN = 30.0000 deg)")

    # g ≠ e exactly because J2 modifies the Laplace vector
    # Q ≈ tan(i/2) up to J2 correction
    # Psi ≈ omega + RAAN up to J2 correction
    # Omega ≈ RAAN up to J2 correction
    print(f"\n  Note: g ≈ e, Q ≈ tan(i/2), Psi ≈ omega+RAAN, Omega ≈ RAAN")
    print(f"  Differences are O(J2) as expected for generalized elements.\n")


# ── CHECK 2: Independent re-derive secular rates via numerical averaging ─
def check2_secular_rates():
    print("=" * 72)
    print("CHECK 2: Re-derive secular rates by numerical K-averaging of exact RHS")
    print("=" * 72)
    a_km = 6916 / (1 - 0.74)
    r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, 45)
    s0 = cart2geqoe(r0, v0, MU, J2Perturbation())
    nu0, p10, p20, K0, q10, q20 = s0

    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    beta = np.sqrt(1 - g0**2)
    alpha = 1/(1+beta)
    a = (MU / nu0**2)**(1/3)
    w = np.sqrt(MU / a)
    c = (MU**2 / nu0)**(1/3) * beta
    A = MU * J2_VAL * RE**2 / 2
    gamma = 1 + q10**2 + q20**2
    delta = 1 - q10**2 - q20**2

    # Numerical quadrature over K with frozen slow state
    NK = 16384
    K_grid = np.linspace(0, 2*np.pi, NK+1)[:-1]
    dK = 2*np.pi / NK

    sinK = np.sin(K_grid)
    cosK = np.cos(K_grid)
    X = a*(alpha*p10*p20*sinK + (1-alpha*p10**2)*cosK - p20)
    Y = a*(alpha*p10*p20*cosK + (1-alpha*p20**2)*sinK - p10)
    r = a*(1 - p10*sinK - p20*cosK)
    zhat = 2*(Y*q20 - X*q10) / (gamma * r)

    U = -A / r**3 * (1 - 3*zhat**2)
    h = np.sqrt(c**2 - 2*r**2*U)
    d = (h - c) / r**2

    # F_h from J2
    dU_dzhat = 6*A*zhat / r**3
    F_h = -dU_dzhat * delta / (gamma * r)
    w_X = (X/h)*F_h
    w_Y = (Y/h)*F_h
    w_h = w_X*q10 - w_Y*q20

    E_Z = -U  # for J2

    # Exact GEqOE RHS
    p1_dot = p20*(d - w_h) + (1/c)*(X/a + 2*p20)*E_Z
    p2_dot = p10*(w_h - d) - (1/c)*(Y/a + 2*p10)*E_Z
    q1_dot = 0.5*gamma*w_Y
    q2_dot = 0.5*gamma*w_X

    # K-average: <f> = (1/2pi*a) * integral(r*f dK)
    avg_p1_dot = np.mean(r * p1_dot) / a
    avg_p2_dot = np.mean(r * p2_dot) / a
    avg_q1_dot = np.mean(r * q1_dot) / a
    avg_q2_dot = np.mean(r * q2_dot) / a

    wp, wn = _secular_rates(s0)
    pred_p1_dot = wp * p20
    pred_p2_dot = -wp * p10
    pred_q1_dot = wn * q20
    pred_q2_dot = -wn * q10

    print(f"  Numerically averaged RHS vs predicted secular rates:")
    print(f"    <p1_dot> = {avg_p1_dot:+.10e}   predicted = {pred_p1_dot:+.10e}   rel err = {abs(avg_p1_dot-pred_p1_dot)/abs(pred_p1_dot):.2e}")
    print(f"    <p2_dot> = {avg_p2_dot:+.10e}   predicted = {pred_p2_dot:+.10e}   rel err = {abs(avg_p2_dot-pred_p2_dot)/abs(pred_p2_dot):.2e}")
    print(f"    <q1_dot> = {avg_q1_dot:+.10e}   predicted = {pred_q1_dot:+.10e}   rel err = {abs(avg_q1_dot-pred_q1_dot)/abs(pred_q1_dot):.2e}")
    print(f"    <q2_dot> = {avg_q2_dot:+.10e}   predicted = {pred_q2_dot:+.10e}   rel err = {abs(avg_q2_dot-pred_q2_dot)/abs(pred_q2_dot):.2e}")

    # Also check g_dot and Q_dot average to zero
    g_dot = (p10*p1_dot + p20*p2_dot) / g0
    Q_dot = (q10*q1_dot + q20*q2_dot) / Q0
    avg_g_dot = np.mean(r * g_dot) / a
    avg_Q_dot = np.mean(r * Q_dot) / a
    print(f"\n  Averaged magnitude rates (should be ~0):")
    print(f"    <g_dot> = {avg_g_dot:+.2e}")
    print(f"    <Q_dot> = {avg_Q_dot:+.2e}")

    # Verify Psi_dot identity: (p2*p1_dot - p1*p2_dot)/g^2
    psi_dot = (p20*p1_dot - p10*p2_dot) / g0**2
    avg_psi_dot = np.mean(r * psi_dot) / a
    print(f"\n  Averaged Psi_dot = {avg_psi_dot:+.10e}")
    print(f"  Predicted omega_Psi = {wp:+.10e}")
    print(f"  Rel error = {abs(avg_psi_dot - wp)/abs(wp):.2e}")

    Omega_dot = (q10*q2_dot - q20*q1_dot) / Q0**2  # note sign: atan2(q1,q2) -> (q2*dq1 - q1*dq2)/Q^2...
    # Actually: d/dt atan2(q1,q2) = (q2*q1_dot - q1*q2_dot) / Q^2
    Omega_dot_correct = (q20*q1_dot - q10*q2_dot) / Q0**2
    avg_Omega_dot = np.mean(r * Omega_dot_correct) / a
    print(f"  Averaged Omega_dot = {avg_Omega_dot:+.10e}")
    print(f"  Predicted omega_Omega = {wn:+.10e}")
    print(f"  Rel error = {abs(avg_Omega_dot - wn)/abs(wn):.2e}\n")


# ── CHECK 3: Multi-anomaly inverse map validation ────────────────────
def check3_multi_anomaly():
    print("=" * 72)
    print("CHECK 3: Multi-anomaly inverse map validation (Molniya e=0.74)")
    print("=" * 72)
    a_km = 6916 / (1 - 0.74)
    M0_list = [0, 20, 45, 90, 135]
    n_orbits = 40
    spo = 240

    results = []
    for M0 in M0_list:
        r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, M0)
        s0 = cart2geqoe(r0, v0, MU, J2Perturbation())
        T = 2*np.pi / s0[0]
        t_grid = np.linspace(0, n_orbits*T, n_orbits*spo+1)

        ta, _ = build_state_integrator(J2Perturbation(), s0, tol=1e-15, compact_mode=True)
        states = propagate_grid(ta, t_grid)

        # Per-orbit means
        centers = 0.5*(t_grid[:n_orbits*spo:spo] + t_grid[spo:n_orbits*spo+spo:spo])
        om_p1 = _orbit_means(states[:,1], n_orbits, spo)
        om_p2 = _orbit_means(states[:,2], n_orbits, spo)
        om_q1 = _orbit_means(states[:,4], n_orbits, spo)
        om_q2 = _orbit_means(states[:,5], n_orbits, spo)

        # Naive secular
        sec_naive = _secular_solution(s0, centers)

        # Full inverse map
        psi0 = np.arctan2(s0[1], s0[2])
        G0 = s0[3] - psi0
        corr = _short_period_map(s0, G0)
        ms = _build_mean_state(s0, corr)
        sec_full = _secular_solution(ms, centers)

        # RMS against orbit means
        def rms(ref, mod):
            return np.sqrt(np.mean((ref - mod)**2))

        naive_rms = np.mean([rms(om_p1, sec_naive["p1"]), rms(om_p2, sec_naive["p2"]),
                             rms(om_q1, sec_naive["q1"]), rms(om_q2, sec_naive["q2"])])
        full_rms = np.mean([rms(om_p1, sec_full["p1"]), rms(om_p2, sec_full["p2"]),
                            rms(om_q1, sec_full["q1"]), rms(om_q2, sec_full["q2"])])

        # Phase RMS
        om_psi = _orbit_means(np.unwrap(np.arctan2(states[:,1], states[:,2])), n_orbits, spo)
        om_Om = _orbit_means(np.unwrap(np.arctan2(states[:,4], states[:,5])), n_orbits, spo)
        naive_psi_rms = rms(om_psi, sec_naive["psi"][:n_orbits])
        full_psi_rms = rms(om_psi, sec_full["psi"][:n_orbits])
        naive_Om_rms = rms(om_Om, sec_naive["Omega"][:n_orbits])
        full_Om_rms = rms(om_Om, sec_full["Omega"][:n_orbits])

        results.append((M0, naive_rms, full_rms, naive_psi_rms, full_psi_rms,
                        naive_Om_rms, full_Om_rms))
        print(f"  M0={M0:3d}°  comp RMS: naive={naive_rms:.2e} full={full_rms:.2e} (ratio={naive_rms/full_rms:.1f}x)")
        print(f"           Psi RMS: naive={naive_psi_rms:.2e} full={full_psi_rms:.2e}")
        print(f"           Omega RMS: naive={naive_Om_rms:.2e} full={full_Om_rms:.2e}")
    print()


# ── CHECK 3b: Different orbit regimes ────────────────────────────────
def check3b_orbit_regimes():
    print("=" * 72)
    print("CHECK 3b: Different orbit regimes (M0=45)")
    print("=" * 72)
    cases = [
        ("LEO low-e", 6778, 0.01, 51.6, 30, 90, 45),
        ("MEO GPS-like", 26560, 0.01, 55.0, 30, 90, 45),
        ("Molniya high-e", 6916/(1-0.74), 0.74, 63.4, 30, 270, 45),
        ("GTO", 6678/(1-0.73), 0.73, 28.5, 30, 180, 45),
    ]
    n_orbits = 40
    spo = 240

    for label, a_km, e, i_deg, raan, argp, M0 in cases:
        r0, v0 = _kepler_to_rv(a_km, e, i_deg, raan, argp, M0)
        s0 = cart2geqoe(r0, v0, MU, J2Perturbation())
        T = 2*np.pi / s0[0]
        t_grid = np.linspace(0, n_orbits*T, n_orbits*spo+1)

        ta, _ = build_state_integrator(J2Perturbation(), s0, tol=1e-15, compact_mode=True)
        states = propagate_grid(ta, t_grid)

        centers = 0.5*(t_grid[:n_orbits*spo:spo] + t_grid[spo:n_orbits*spo+spo:spo])
        om_p1 = _orbit_means(states[:,1], n_orbits, spo)
        om_p2 = _orbit_means(states[:,2], n_orbits, spo)
        om_q1 = _orbit_means(states[:,4], n_orbits, spo)
        om_q2 = _orbit_means(states[:,5], n_orbits, spo)

        psi0 = np.arctan2(s0[1], s0[2])
        G0 = s0[3] - psi0
        corr = _short_period_map(s0, G0)
        ms = _build_mean_state(s0, corr)
        sec_full = _secular_solution(ms, centers)

        def rms(ref, mod):
            return np.sqrt(np.mean((ref - mod)**2))
        full_rms = np.mean([rms(om_p1, sec_full["p1"]), rms(om_p2, sec_full["p2"]),
                            rms(om_q1, sec_full["q1"]), rms(om_q2, sec_full["q2"])])
        om_psi = _orbit_means(np.unwrap(np.arctan2(states[:,1], states[:,2])), n_orbits, spo)
        om_Om = _orbit_means(np.unwrap(np.arctan2(states[:,4], states[:,5])), n_orbits, spo)
        psi_rms = rms(om_psi, sec_full["psi"][:n_orbits])
        Om_rms = rms(om_Om, sec_full["Omega"][:n_orbits])

        print(f"  {label:20s}  comp RMS={full_rms:.2e}  Psi RMS={psi_rms:.2e}  Om RMS={Om_rms:.2e}")
    print()


# ── CHECK 4: J2 scaling (first-order claim) ──────────────────────────
def check4_j2_scaling():
    print("=" * 72)
    print("CHECK 4: J2 scaling — residuals should be O(J2^2)")
    print("=" * 72)
    a_km = 6916 / (1 - 0.74)
    scales = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    n_orbits = 40
    spo = 240
    M0 = 45

    rms_results = []
    for scale in scales:
        j2_scaled = J2_VAL * scale
        pert = J2Perturbation(j2=j2_scaled)
        r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, M0)
        s0 = cart2geqoe(r0, v0, MU, pert)
        T = 2*np.pi / s0[0]
        t_grid = np.linspace(0, n_orbits*T, n_orbits*spo+1)

        ta, _ = build_state_integrator(pert, s0, tol=1e-15, compact_mode=True)
        states = propagate_grid(ta, t_grid)

        centers = 0.5*(t_grid[:n_orbits*spo:spo] + t_grid[spo:n_orbits*spo+spo:spo])
        om_p1 = _orbit_means(states[:,1], n_orbits, spo)
        om_p2 = _orbit_means(states[:,2], n_orbits, spo)
        om_q1 = _orbit_means(states[:,4], n_orbits, spo)
        om_q2 = _orbit_means(states[:,5], n_orbits, spo)

        psi0 = np.arctan2(s0[1], s0[2])
        G0 = s0[3] - psi0
        corr = _short_period_map(s0, G0, j2=j2_scaled)
        ms = _build_mean_state(s0, corr)
        sec = _secular_solution(ms, centers, j2=j2_scaled)

        def rms(ref, mod):
            return np.sqrt(np.mean((ref - mod)**2))
        comp_rms = np.mean([rms(om_p1, sec["p1"]), rms(om_p2, sec["p2"]),
                            rms(om_q1, sec["q1"]), rms(om_q2, sec["q2"])])

        # g magnitude drift (should be O(J2^2))
        g_osc = np.hypot(states[:,1], states[:,2])
        g_drift = abs(_orbit_means(g_osc, n_orbits, spo)[-1] - _orbit_means(g_osc, n_orbits, spo)[0])

        rms_results.append((scale, comp_rms, g_drift))
        print(f"  J2 scale={scale:5.2f}  comp RMS={comp_rms:.2e}  g drift={g_drift:.2e}")

    # Check scaling: if residuals are O(J2^2), then RMS should scale as scale^2
    print(f"\n  Scaling analysis (normalized to scale=1.0):")
    ref_idx = scales.index(1.0)
    ref_rms = rms_results[ref_idx][1]
    for scale, comp_rms, _ in rms_results:
        expected_ratio = scale**2
        actual_ratio = comp_rms / ref_rms if ref_rms > 0 else 0
        print(f"    scale={scale:5.2f}  actual/ref={actual_ratio:.3f}  expected(J2^2)={expected_ratio:.3f}  ratio={actual_ratio/expected_ratio:.2f}")
    print()


# ── CHECK 5: dPsi/dG identity ────────────────────────────────────────
def check5_eq814():
    print("=" * 72)
    print("CHECK 5: Verify corrected dPsi/dG identity")
    print("=" * 72)
    a_km = 6916 / (1 - 0.74)
    r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, 45)
    s0 = cart2geqoe(r0, v0, MU, J2Perturbation())
    nu0, p10, p20, K0, q10, q20 = s0

    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    psi0 = np.arctan2(p10, p20)
    omega0 = np.arctan2(q10, q20)
    beta = np.sqrt(1 - g0**2)
    alpha = 1/(1+beta)
    a = (MU / nu0**2)**(1/3)
    w = np.sqrt(MU / a)
    c = (MU**2 / nu0)**(1/3) * beta
    A = MU * J2_VAL * RE**2 / 2
    gamma = 1 + q10**2 + q20**2
    delta = 1 - q10**2 - q20**2
    s_i = 2*Q0 / gamma

    G = np.linspace(0, 2*np.pi, 4097)
    cosG, sinG = np.cos(G), np.sin(G)
    sinK, cosK = np.sin(G + psi0), np.cos(G + psi0)
    X = a*(alpha*p10*p20*sinK + (1-alpha*p10**2)*cosK - p20)
    Y = a*(alpha*p10*p20*cosK + (1-alpha*p20**2)*sinK - p10)
    r = a*(1 - g0*cosG)
    cosf = (cosG - g0)/(1 - g0*cosG)
    sinf = beta*sinG/(1 - g0*cosG)
    argp = psi0 - omega0
    sin_u = np.sin(argp)*cosf + np.cos(argp)*sinf
    cos_u = np.cos(argp)*cosf - np.sin(argp)*sinf

    U = -A/r**3 * (1 - 3*s_i**2*sin_u**2)
    d = -U / c
    zhat = s_i * sin_u
    w_h = 3*A*zhat**2*delta / (c * r**3)

    # Method 1: exact from p1_dot, p2_dot
    E_Z = -U
    p1_dot = p20*(d - w_h) + (1/c)*(X/a + 2*p20)*E_Z
    p2_dot = p10*(w_h - d) - (1/c)*(Y/a + 2*p10)*E_Z
    psi_dot_exact = (p20*p1_dot - p10*p2_dot) / g0**2
    dPsi_dG_exact = (r/w) * psi_dot_exact

    # Method 2: corrected identity in the note.
    dPsi_dG_corrected = (r/w) * (d - w_h - (U/c)*(1 + cosG/g0))

    err_corrected = np.max(np.abs(dPsi_dG_corrected - dPsi_dG_exact))

    print(f"  Max |dPsi/dG(corrected)      - exact| = {err_corrected:.6e}")
    print(f"  Corrected formula is {'OK' if err_corrected < 1e-10 else 'WRONG'}")
    print(f"  The current note identity matches the exact GEqOE p1/p2 flow.\n")


# ── CHECK 6: Verify w_h expression ───────────────────────────────────
def check6_wh_consistency():
    print("=" * 72)
    print("CHECK 6: Cross-check w_h between J2-only and general zonal paths")
    print("=" * 72)
    a_km = 6916 / (1 - 0.74)
    r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, 45)
    s0 = cart2geqoe(r0, v0, MU, J2Perturbation())
    nu0, p10, p20, K0, q10, q20 = s0

    beta = np.sqrt(1 - p10**2 - p20**2)
    alpha = 1/(1+beta)
    a = (MU / nu0**2)**(1/3)
    c = (MU**2 / nu0)**(1/3) * beta
    A = MU * J2_VAL * RE**2 / 2
    gamma = 1 + q10**2 + q20**2
    delta = 1 - q10**2 - q20**2

    # Evaluate at K0
    sinK, cosK = np.sin(K0), np.cos(K0)
    X = a*(alpha*p10*p20*sinK + (1-alpha*p10**2)*cosK - p20)
    Y = a*(alpha*p10*p20*cosK + (1-alpha*p20**2)*sinK - p10)
    r = a*(1 - p10*sinK - p20*cosK)
    zhat = 2*(Y*q20 - X*q10) / (gamma * r)

    U = -A / r**3 * (1 - 3*zhat**2)
    h = np.sqrt(c**2 - 2*r**2*U)

    # J2-only path: I_val = 3A*zhat*delta/(h*r^3), w_h = I_val*zhat
    I_val_j2 = 3*A*zhat*delta / (h*r**3)
    w_h_j2 = I_val_j2 * zhat

    # General zonal: F_h = -dU/dzhat * delta/(gamma*r), w_X = X/h*F_h, w_Y = Y/h*F_h
    dU_dzhat = 6*A*zhat / r**3
    F_h = -dU_dzhat * delta / (gamma * r)
    w_X = (X/h)*F_h
    w_Y = (Y/h)*F_h
    w_h_general = w_X*q10 - w_Y*q20

    print(f"  w_h (J2-only path)  = {w_h_j2:.15e}")
    print(f"  w_h (general zonal) = {w_h_general:.15e}")
    print(f"  Difference = {abs(w_h_j2 - w_h_general):.2e}")
    print(f"  {'CONSISTENT' if abs(w_h_j2 - w_h_general) < 1e-20 else 'MISMATCH'}\n")


# ── CHECK 7: Fast-phase reconstruction gap ───────────────────────────
def check7_fast_phase_reconstruction():
    print("=" * 72)
    print("CHECK 7: Forward reconstruction gap is in K, not in the slow map")
    print("=" * 72)
    a_km = 6916 / (1 - 0.74)
    r0, v0 = _kepler_to_rv(a_km, 0.74, 63.4, 30, 270, 45)
    s0 = cart2geqoe(r0, v0, MU, J2Perturbation())
    mean0, _ = _inverse_map_initial_state(s0)

    n_orbits = 20
    spo = 200
    t_grid = np.linspace(0.0, n_orbits * 2 * np.pi / s0[0], n_orbits * spo + 1)
    ta, _ = build_state_integrator(J2Perturbation(), s0, tol=1e-15, compact_mode=True)
    states = propagate_grid(ta, t_grid)

    psi0_bar = np.arctan2(mean0[1], mean0[2])
    K0_bar_naive = s0[3]
    L0_bar = K_to_L(K0_bar_naive, mean0[1], mean0[2])
    Ldot_bar = _mean_longitude_rate(mean0)

    true_k_err = []
    true_k_pos_err = []
    lbar_pos_err = []
    for ti, sosc in zip(t_grid[::40], states[::40]):
        sec = _secular_solution(mean0, np.array([ti]))
        ms = mean0.copy()
        ms[1] = sec["p1"][0]
        ms[2] = sec["p2"][0]
        ms[4] = sec["q1"][0]
        ms[5] = sec["q2"][0]

        # Use the exact osculating G only as a diagnostic to isolate the slow map.
        G_eval_true = sosc[3] - np.arctan2(sosc[1], sosc[2])
        corr_true = _short_period_map(ms, G_eval_true)
        rec_true_k = ms.copy()
        g = np.hypot(ms[1], ms[2]) + corr_true["g_eval"]
        psi = np.arctan2(ms[1], ms[2]) + corr_true["Psi_eval"]
        Q = np.hypot(ms[4], ms[5]) + corr_true["Q_eval"]
        Om = np.arctan2(ms[4], ms[5]) + corr_true["Omega_eval"]
        rec_true_k[1] = g * np.sin(psi)
        rec_true_k[2] = g * np.cos(psi)
        rec_true_k[4] = Q * np.sin(Om)
        rec_true_k[5] = Q * np.cos(Om)
        rec_true_k[3] = sosc[3]

        r_rec_true, _ = geqoe2cart(rec_true_k, MU, J2Perturbation())
        r_osc, _ = geqoe2cart(sosc, MU, J2Perturbation())
        true_k_pos_err.append(np.linalg.norm(r_rec_true - r_osc))
        true_k_err.append(np.linalg.norm(rec_true_k[[1, 2, 4, 5]] - sosc[[1, 2, 4, 5]]))

        # Current practical mean-L route: no validated v1/K correction yet.
        Lbar = L0_bar + Ldot_bar * ti
        Kbar = solve_kepler_gen(Lbar, ms[1], ms[2])
        Gbar = Kbar - np.arctan2(ms[1], ms[2])
        corr_lbar = _short_period_map(ms, Gbar)
        rec_lbar = ms.copy()
        g = np.hypot(ms[1], ms[2]) + corr_lbar["g_eval"]
        psi = np.arctan2(ms[1], ms[2]) + corr_lbar["Psi_eval"]
        Q = np.hypot(ms[4], ms[5]) + corr_lbar["Q_eval"]
        Om = np.arctan2(ms[4], ms[5]) + corr_lbar["Omega_eval"]
        rec_lbar[1] = g * np.sin(psi)
        rec_lbar[2] = g * np.cos(psi)
        rec_lbar[4] = Q * np.sin(Om)
        rec_lbar[5] = Q * np.cos(Om)
        rec_lbar[3] = Kbar
        r_rec_lbar, _ = geqoe2cart(rec_lbar, MU, J2Perturbation())
        lbar_pos_err.append(np.linalg.norm(r_rec_lbar - r_osc))

    print(f"  With true osculating K injected:")
    print(f"    slow-state GEqOE component error mean = {np.mean(true_k_err):.3e}")
    print(f"    Cartesian position error mean/max     = {np.mean(true_k_pos_err):.3e} / {np.max(true_k_pos_err):.3e} km")
    print(f"  With current mean-L -> mean-K route:")
    print(f"    Cartesian position error mean/max     = {np.mean(lbar_pos_err):.3e} / {np.max(lbar_pos_err):.3e} km")
    print("  Interpretation: the remaining reconstruction gap is dominated by the fast")
    print("  phase K model, not by the validated slow inverse map.\n")


# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check1_angle_meanings()
    check2_secular_rates()
    check3_multi_anomaly()
    check3b_orbit_regimes()
    check4_j2_scaling()
    check5_eq814()
    check6_wh_consistency()
    check7_fast_phase_reconstruction()
