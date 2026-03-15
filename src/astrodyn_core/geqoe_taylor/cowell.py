"""Cowell (Cartesian) propagators for ground truth reference.

Implementations:
  - propagate_cowell: scipy DOP853 (J2-only, no extra dependencies)
  - propagate_cowell_heyoka: heyoka Taylor (J2-only, highest accuracy)
  - propagate_cowell_heyoka_full: heyoka Taylor (J2 + Sun + Moon)
  - propagate_cowell_heyoka_general: heyoka Taylor for arbitrary U/P models,
    optionally with propagated mass
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from astrodyn_core.geqoe_taylor.constants import MU, A_J2


def _parameter_defaults(perturbation) -> dict[str, float]:
    defaults = getattr(perturbation, "parameter_defaults", None)
    if defaults is None:
        return {}
    return dict(defaults())


def _build_par_values(perturbation, par_map):
    if not par_map:
        return []
    n_pars = max(par_map.values()) + 1
    par_values = [0.0] * n_pars
    if "mu" in par_map:
        par_values[par_map["mu"]] = getattr(perturbation, "mu", MU)
    for name, value in _parameter_defaults(perturbation).items():
        if name in par_map:
            par_values[par_map[name]] = float(value)
    return par_values


def _cowell_rhs(t: float, y: np.ndarray, mu: float, A: float) -> np.ndarray:
    """Cartesian equations of motion with J2 (for scipy)."""
    r_vec = y[:3]
    v_vec = y[3:]

    r = np.linalg.norm(r_vec)
    r2 = r * r
    r5 = r2 * r2 * r
    x, y_c, z = r_vec
    zhat2 = (z / r) ** 2

    mu_r3 = mu / (r2 * r)
    coeff = 3.0 * A / r5
    factor_xy = 1.0 - 5.0 * zhat2
    factor_z = 3.0 - 5.0 * zhat2

    ax = -mu_r3 * x - coeff * x * factor_xy
    ay = -mu_r3 * y_c - coeff * y_c * factor_xy
    az = -mu_r3 * z - coeff * z * factor_z

    return np.array([v_vec[0], v_vec[1], v_vec[2], ax, ay, az])


def propagate_cowell(
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    mu: float = MU,
    A: float = A_J2,
    rtol: float = 1e-14,
    atol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate Cartesian state with J2 using DOP853."""
    y0 = np.concatenate([r0, v0])
    sol = solve_ivp(
        _cowell_rhs,
        [0.0, t_final],
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        args=(mu, A),
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Cowell integration failed: {sol.message}")

    return sol.y[:3, -1], sol.y[3:, -1]


def _build_cowell_heyoka_system(mu_val: float, A_val: float):
    """Build heyoka Cowell ODE system for J2."""
    import heyoka as hy

    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    r2 = x * x + y * y + z * z
    r = hy.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3
    zhat2 = (z * z) / r2

    mu = mu_val
    A = A_val
    mu_r3 = mu / r3
    coeff = 3.0 * A / r5
    fxy = 1.0 - 5.0 * zhat2
    fz = 3.0 - 5.0 * zhat2

    ax = -mu_r3 * x - coeff * x * fxy
    ay = -mu_r3 * y - coeff * y * fxy
    az = -mu_r3 * z - coeff * z * fz

    return [
        (x, vx), (y, vy), (z, vz),
        (vx, ax), (vy, ay), (vz, az),
    ]


def propagate_cowell_heyoka(
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    mu: float = MU,
    A: float = A_J2,
    tol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate Cartesian state with J2 using heyoka Taylor integrator."""
    import heyoka as hy

    sys = _build_cowell_heyoka_system(mu, A)
    ic = list(r0) + list(v0)

    ta = hy.taylor_adaptive(sys, ic, tol=tol)
    ta.propagate_until(t_final)

    return ta.state[:3].copy(), ta.state[3:].copy()


def _build_cowell_heyoka_general_system(
    perturbation,
    mu_val: float | None = None,
    use_par: bool = True,
    time_origin: float = 0.0,
    with_mass: bool = False,
):
    """Build a general Cartesian Cowell system from the perturbation interface."""
    import heyoka as hy

    if with_mass:
        x, y, z, vx, vy, vz, m = hy.make_vars("x", "y", "z", "vx", "vy", "vz", "m")
        state_vars = [x, y, z, vx, vy, vz, m]
    else:
        x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
        state_vars = [x, y, z, vx, vy, vz]
        m = None

    if use_par:
        mu = hy.par[0]
        par_map = {"mu": 0}
        pars = {"mu": mu}
        for idx, (name, _) in enumerate(_parameter_defaults(perturbation).items(), start=1):
            par_map[name] = idx
            pars[name] = hy.par[idx]
    else:
        mu = float(mu_val if mu_val is not None else MU)
        par_map = {}
        pars = {"mu": mu}
        for name, value in _parameter_defaults(perturbation).items():
            pars[name] = float(value)

    if with_mass:
        pars["mass"] = m

    r2 = x * x + y * y + z * z
    r = hy.sqrt(r2)
    r3 = r2 * r
    t_expr = hy.time - float(time_origin)

    dUdx, dUdy, dUdz = perturbation.grad_U_expr(x, y, z, r, t_expr, pars)
    if with_mass and hasattr(perturbation, "P_and_mass_flow_expr"):
        Px, Py, Pz, m_dot = perturbation.P_and_mass_flow_expr(
            x, y, z, vx, vy, vz, r, t_expr, pars
        )
    else:
        Px, Py, Pz = perturbation.P_expr(x, y, z, vx, vy, vz, r, t_expr, pars)
        if with_mass and hasattr(perturbation, "mass_flow_expr"):
            m_dot = perturbation.mass_flow_expr(
                x, y, z, vx, vy, vz, r, t_expr, pars
            )
        elif with_mass:
            m_dot = 0.0 * m

    ax = -mu * x / r3 + Px - dUdx
    ay = -mu * y / r3 + Py - dUdy
    az = -mu * z / r3 + Pz - dUdz

    sys = [
        (x, vx), (y, vy), (z, vz),
        (vx, ax), (vy, ay), (vz, az),
    ]
    if with_mass:
        sys.append((m, m_dot))

    return sys, state_vars, par_map


def propagate_cowell_heyoka_general(
    perturbation,
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    t0: float = 0.0,
    m0: float | None = None,
    mu: float = MU,
    tol: float = 1e-15,
    compact_mode: bool = True,
):
    """Propagate a Cartesian state with a general perturbation model via heyoka.

    Args:
        perturbation: model providing ``grad_U_expr`` and ``P_expr``.
        r0: initial position in km.
        v0: initial velocity in km/s.
        t_final: target integrator time in seconds.
        t0: initial integrator time in seconds.
        m0: optional initial mass in kg. If provided, a 7th mass state is
            propagated using ``mass_flow_expr``.
        mu: central-body gravitational parameter.
        tol: Taylor tolerance.
        compact_mode: heyoka compact-mode toggle.

    Returns:
        ``(r, v)`` for the 6-state case, or ``(r, v, m)`` when ``m0`` is set.
    """
    import heyoka as hy

    with_mass = m0 is not None
    sys, _, par_map = _build_cowell_heyoka_general_system(
        perturbation,
        mu_val=mu,
        use_par=True,
        time_origin=t0,
        with_mass=with_mass,
    )
    ic = list(r0) + list(v0)
    if with_mass:
        ic.append(float(m0))

    ta = hy.taylor_adaptive(
        sys,
        state=ic,
        time=t0,
        pars=_build_par_values(perturbation, par_map),
        tol=tol,
        compact_mode=compact_mode,
    )
    ta.propagate_until(t_final)

    if with_mass:
        return ta.state[:3].copy(), ta.state[3:6].copy(), float(ta.state[6])
    return ta.state[:3].copy(), ta.state[3:6].copy()


def _build_cowell_heyoka_full_system(
    mu_val: float,
    A_val: float,
    epoch_jd: float,
    include_sun: bool = True,
    include_moon: bool = True,
    sun_thresh: float = 1e-9,
    moon_thresh: float = 1e-6,
):
    """Build heyoka Cowell system: J2 + Sun + Moon."""
    import heyoka as hy
    from astrodyn_core.geqoe_taylor.constants import (
        AU_KM, COS_OBLIQUITY, SIN_OBLIQUITY,
        GM_SUN, GM_MOON, JD_J2000,
        SECONDS_PER_DAY, DAYS_PER_JULIAN_CENTURY, DAYS_PER_JULIAN_MILLENNIUM,
    )

    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    r2 = x * x + y * y + z * z
    r = hy.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3
    zhat2 = (z * z) / r2

    mu = mu_val
    A = A_val

    # J2 acceleration
    mu_r3 = mu / r3
    coeff = 3.0 * A / r5
    fxy = 1.0 - 5.0 * zhat2
    fz = 3.0 - 5.0 * zhat2

    ax = -mu_r3 * x - coeff * x * fxy
    ay = -mu_r3 * y - coeff * y * fxy
    az = -mu_r3 * z - coeff * z * fz

    # Third-body accelerations
    def _add_third_body(ax, ay, az, x3_eq, y3_eq, z3_eq, mu_3b):
        dx = x3_eq - x
        dy = y3_eq - y
        dz = z3_eq - z
        d2 = dx * dx + dy * dy + dz * dz
        d_mag = hy.sqrt(d2)
        d3 = d2 * d_mag

        r3_2 = x3_eq * x3_eq + y3_eq * y3_eq + z3_eq * z3_eq
        r3_mag = hy.sqrt(r3_2)
        r3_3 = r3_2 * r3_mag

        ax = ax + mu_3b * (dx / d3 - x3_eq / r3_3)
        ay = ay + mu_3b * (dy / d3 - y3_eq / r3_3)
        az = az + mu_3b * (dz / d3 - z3_eq / r3_3)
        return ax, ay, az

    def _ecl2eq(x_ecl, y_ecl, z_ecl):
        return (x_ecl,
                y_ecl * COS_OBLIQUITY - z_ecl * SIN_OBLIQUITY,
                y_ecl * SIN_OBLIQUITY + z_ecl * COS_OBLIQUITY)

    if include_sun:
        t_jm = (epoch_jd - JD_J2000) / DAYS_PER_JULIAN_MILLENNIUM + (
            hy.time / (SECONDS_PER_DAY * DAYS_PER_JULIAN_MILLENNIUM)
        )
        emb = hy.model.vsop2013_cartesian(3, time_expr=t_jm, thresh=sun_thresh)
        sun_ecl = [-emb[i] * AU_KM for i in range(3)]
        sun_eq = _ecl2eq(*sun_ecl)
        ax, ay, az = _add_third_body(ax, ay, az, *sun_eq, GM_SUN)

    if include_moon:
        t_jc = (epoch_jd - JD_J2000) / DAYS_PER_JULIAN_CENTURY + (
            hy.time / (SECONDS_PER_DAY * DAYS_PER_JULIAN_CENTURY)
        )
        moon_ecl = hy.model.elp2000_cartesian_e2000(time_expr=t_jc, thresh=moon_thresh)
        moon_eq = _ecl2eq(moon_ecl[0], moon_ecl[1], moon_ecl[2])
        ax, ay, az = _add_third_body(ax, ay, az, *moon_eq, GM_MOON)

    return [
        (x, vx), (y, vy), (z, vz),
        (vx, ax), (vy, ay), (vz, az),
    ]


def propagate_cowell_heyoka_full(
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    epoch_jd: float,
    mu: float = MU,
    A: float = A_J2,
    tol: float = 1e-15,
    include_sun: bool = True,
    include_moon: bool = True,
    sun_thresh: float = 1e-9,
    moon_thresh: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate Cartesian state with J2 + Sun + Moon using heyoka Taylor.

    Args:
        r0: initial position (3,) in km (equatorial J2000).
        v0: initial velocity (3,) in km/s.
        t_final: propagation duration in seconds.
        epoch_jd: Julian date of epoch (TDB).
        mu: gravitational parameter.
        A: J2 convenience constant.
        tol: Taylor integrator tolerance.
        include_sun: include solar third-body.
        include_moon: include lunar third-body.
        sun_thresh: VSOP2013 truncation threshold.
        moon_thresh: ELP2000 truncation threshold.

    Returns:
        (r_final, v_final): final position and velocity.
    """
    import heyoka as hy

    sys = _build_cowell_heyoka_full_system(
        mu, A, epoch_jd,
        include_sun=include_sun, include_moon=include_moon,
        sun_thresh=sun_thresh, moon_thresh=moon_thresh,
    )
    ic = list(r0) + list(v0)

    ta = hy.taylor_adaptive(sys, ic, tol=tol)
    ta.propagate_until(t_final)

    return ta.state[:3].copy(), ta.state[3:].copy()
