"""GEqOE symbolic RHS construction for heyoka.

Builds the GEqOE equations of motion as heyoka symbolic expressions.
State: [nu, p1, p2, K, q1, q2].

Three code paths (selected automatically):
  - J2-only (Section 7.1): simplified equations with E_dot=0, 2U-rFr=-U
  - Zonal fast path: like J2-only but with general zonal potential, Euler
    identity for 2U-rFr, and F_h from dU/dzhat. No Cartesian detour.
  - General (Eqs. 45-51): full equations for arbitrary perturbations

Reference: Bau, Hernando-Ayuso & Bombardelli (2021), Celest. Mech. Dyn. Astr. 133:50.
"""

from __future__ import annotations

import heyoka as hy

from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel


def _can_use_zonal_path(perturbation) -> bool:
    """Check if perturbation supports the optimized zonal fast path."""
    return (
        getattr(perturbation, "_zonal_fast_path", False)
        and getattr(perturbation, "is_conservative", True)
        and not getattr(perturbation, "is_time_dependent", False)
    )


def _is_j2_only(perturbation) -> bool:
    """Check if perturbation explicitly supports the pure-J2 fast path.

    The J2 fast path hard-codes the Eq. 56 potential and Section 7.1
    simplifications, so it must only be enabled by models that are known to
    represent that exact problem.
    """
    return getattr(perturbation, "_j2_fast_path", False)


def build_geqoe_system(
    perturbation: PerturbationModel,
    mu_val: float | None = None,
    use_par: bool = True,
    time_origin: float = 0.0,
):
    """Build the GEqOE ODE system as heyoka expressions.

    Auto-detects the best code path:
      1. Zonal fast path (if perturbation has zonal_quantities)
      2. J2-only fast path (if perturbation opts into _j2_fast_path)
      3. General path (for arbitrary perturbations)

    Args:
        perturbation: PerturbationModel providing U_expr (and optionally
                      grad_U_expr, P_expr for general perturbations).
        mu_val: gravitational parameter (km^3/s^2). Required if use_par=False.
        use_par: if True, mu is a runtime parameter (hy.par[0]).
        time_origin: absolute heyoka time corresponding to t=0 for
                     time-dependent perturbation models.

    Returns:
        sys: list of (var, rhs) tuples for heyoka.taylor_adaptive.
        state_vars: list of 6 heyoka variable expressions.
        par_map: dict mapping parameter names to par[] indices.
    """
    if _can_use_zonal_path(perturbation):
        return _build_zonal_system(perturbation, mu_val, use_par)
    elif _is_j2_only(perturbation):
        return _build_j2_only_system(perturbation, mu_val, use_par)
    else:
        return _build_general_system(
            perturbation, mu_val, use_par, time_origin=time_origin
        )


def _build_intermediates(mu, nu, p1, p2, K, q1, q2):
    """Build intermediate quantities shared by all code paths.

    Returns dict of named heyoka expressions.
    """
    sinK = hy.sin(K)
    cosK = hy.cos(K)

    a = (mu / (nu * nu)) ** (1.0 / 3.0)
    g2 = p1 * p1 + p2 * p2
    beta = hy.sqrt(1.0 - g2)
    alpha = 1.0 / (1.0 + beta)

    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1 * p1) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2 * p2) * sinK - p1)

    r = a * (1.0 - p1 * sinK - p2 * cosK)

    cosL = X / r
    sinL = Y / r

    gamma = 1.0 + q1 * q1 + q2 * q2
    gamma_inv = 1.0 / gamma
    zhat = 2.0 * (Y * q2 - X * q1) * gamma_inv / r

    c = (mu * mu / nu) ** (1.0 / 3.0) * beta
    w = hy.sqrt(mu / a)

    # rdot from Eq. 32
    rdot = w * a / r * (p2 * sinK - p1 * cosK)

    return dict(
        sinK=sinK, cosK=cosK, a=a, g2=g2, beta=beta, alpha=alpha,
        X=X, Y=Y, r=r, cosL=cosL, sinL=sinL, gamma=gamma,
        gamma_inv=gamma_inv, zhat=zhat, c=c, w=w, rdot=rdot,
    )


def _build_j2_only_system(perturbation, mu_val, use_par):
    """Build the J2-only GEqOE ODE system (Section 7.1).

    Simplifications: E_dot=0, nu_dot=0, 2U-rFr=-U.
    """
    nu, p1, p2, K, q1, q2 = hy.make_vars("nu", "p1", "p2", "K", "q1", "q2")
    state_vars = [nu, p1, p2, K, q1, q2]

    if use_par:
        mu = hy.par[0]
        A = hy.par[1]
        par_map = {"mu": 0, "A_J2": 1}
    else:
        from astrodyn_core.geqoe_taylor.constants import MU, A_J2
        mu = float(mu_val if mu_val is not None else MU)
        A = float(A_J2)
        par_map = {}

    im = _build_intermediates(mu, nu, p1, p2, K, q1, q2)
    r = im["r"]
    X, Y = im["X"], im["Y"]
    a, c, w = im["a"], im["c"], im["w"]
    alpha = im["alpha"]
    sinK, cosK = im["sinK"], im["cosK"]
    cosL, sinL = im["cosL"], im["sinL"]
    zhat = im["zhat"]

    # J2 potential
    r3 = r * r * r
    U_val = -A / r3 * (1.0 - 3.0 * zhat * zhat)

    # Physical angular momentum
    h = hy.sqrt(c * c - 2.0 * r * r * U_val)
    h_minus_c = -2.0 * r * r * U_val / (h + c)

    delta = 1.0 - q1 * q1 - q2 * q2
    I_val = 3.0 * A * zhat * delta / (h * r3)
    w_h = I_val * zhat
    d = h_minus_c / (r * r)

    # --- Assemble ODEs ---
    nu_dot = 0.0 * nu

    xi1 = X / a + 2.0 * p2
    p1_dot = p2 * (d - w_h) - (1.0 / c) * xi1 * U_val

    xi2 = Y / a + 2.0 * p1
    p2_dot = p1 * (w_h - d) + (1.0 / c) * xi2 * U_val

    one_minus_r_over_a = p1 * sinK + p2 * cosK
    K_dot = (w / r + d - w_h
             - (1.0 / c) * (1.0 + alpha * one_minus_r_over_a) * U_val)

    q1_dot = -I_val * sinL
    q2_dot = -I_val * cosL

    sys = [
        (nu, nu_dot), (p1, p1_dot), (p2, p2_dot),
        (K, K_dot), (q1, q1_dot), (q2, q2_dot),
    ]
    return sys, state_vars, par_map


def _build_zonal_system(perturbation, mu_val, use_par):
    """Build the GEqOE ODE system for zonal harmonics (fast path).

    Like the J2-only path but generalized:
      - U, dU/dzhat, and Euler term from perturbation.zonal_quantities()
      - 2U - rFr via Euler homogeneity: sum_n (1-n)*U_n
      - F_h = -(dU/dzhat) * delta / (gamma * r)
        (because position lies in orbital plane: r_vec . eZ = 0)
      - E_dot = 0, nu_dot = 0 (conservative, time-independent)
      - No Cartesian coordinates, frame vectors, or 3D gradient needed
    """
    nu, p1, p2, K, q1, q2 = hy.make_vars("nu", "p1", "p2", "K", "q1", "q2")
    state_vars = [nu, p1, p2, K, q1, q2]

    if use_par:
        mu = hy.par[0]
        par_map = {"mu": 0}
    else:
        from astrodyn_core.geqoe_taylor.constants import MU
        mu = float(mu_val if mu_val is not None else MU)
        par_map = {}

    im = _build_intermediates(mu, nu, p1, p2, K, q1, q2)
    r = im["r"]
    X, Y = im["X"], im["Y"]
    a, c, w = im["a"], im["c"], im["w"]
    alpha = im["alpha"]
    sinK, cosK = im["sinK"], im["cosK"]
    gamma = im["gamma"]
    zhat = im["zhat"]

    # --- Zonal quantities from the perturbation model ---
    U_val, dU_dzhat, euler_val = perturbation.zonal_quantities(r, zhat)

    # Physical angular momentum
    h = hy.sqrt(c * c - 2.0 * r * r * U_val)
    h_minus_c = -2.0 * r * r * U_val / (h + c)
    d = h_minus_c / (r * r)

    # Out-of-plane force: F_h = -(dU/dzhat) * delta / (gamma * r)
    # Derivation: F = -grad(U), F_h = F . eZ.
    # Since r lies in the eX-eY plane, r . eZ = 0, and the chain-rule
    # terms through dr/dx cancel, leaving only the dzhat/dz contribution
    # projected onto eZ_z = delta/gamma.
    delta = 2.0 - gamma   # = 1 - q1^2 - q2^2
    F_h = -dU_dzhat * delta / (gamma * r)

    # Angular rates (Eq. 14, 52)
    w_X = (X / h) * F_h
    w_Y = (Y / h) * F_h
    w_h = w_X * q1 - w_Y * q2

    # --- Assemble ODEs ---
    nu_dot = 0.0 * nu

    # p-dot (Eqs. 47-48 with E_dot=0, using Euler term for 2U-rFr)
    p1_dot = p2 * (d - w_h) + (1.0 / c) * (X / a + 2.0 * p2) * euler_val
    p2_dot = p1 * (w_h - d) - (1.0 / c) * (Y / a + 2.0 * p1) * euler_val

    # K_dot (Eq. 75 with E_dot=0)
    one_minus_r_over_a = p1 * sinK + p2 * cosK
    K_dot = (w / r + d - w_h
             + (1.0 / c) * (1.0 + alpha * one_minus_r_over_a) * euler_val)

    # q-dot (Eqs. 50-51)
    q1_dot = 0.5 * gamma * w_Y
    q2_dot = 0.5 * gamma * w_X

    sys = [
        (nu, nu_dot), (p1, p1_dot), (p2, p2_dot),
        (K, K_dot), (q1, q1_dot), (q2, q2_dot),
    ]
    return sys, state_vars, par_map


def _build_general_system(perturbation, mu_val, use_par, time_origin: float = 0.0):
    """Build the full GEqOE ODE system (Eqs. 45-51, K from L).

    Supports arbitrary conservative (U) and non-conservative (P) perturbations.
    """
    nu, p1, p2, K, q1, q2 = hy.make_vars("nu", "p1", "p2", "K", "q1", "q2")
    state_vars = [nu, p1, p2, K, q1, q2]

    if use_par:
        mu = hy.par[0]
        par_map = {"mu": 0}
    else:
        from astrodyn_core.geqoe_taylor.constants import MU
        mu = float(mu_val if mu_val is not None else MU)
        par_map = {}

    pars = {"mu": mu}

    im = _build_intermediates(mu, nu, p1, p2, K, q1, q2)
    sinK, cosK = im["sinK"], im["cosK"]
    a = im["a"]
    alpha = im["alpha"]
    X, Y = im["X"], im["Y"]
    r = im["r"]
    cosL, sinL = im["cosL"], im["sinL"]
    gamma, gamma_inv = im["gamma"], im["gamma_inv"]
    c, w = im["c"], im["w"]
    rdot = im["rdot"]

    # --- Equinoctial frame vectors (Eq. 37) ---
    q1s = q1 * q1
    q2s = q2 * q2
    q1q2 = q1 * q2

    eX = [gamma_inv * (1.0 - q1s + q2s),
          gamma_inv * (2.0 * q1q2),
          gamma_inv * (-2.0 * q1)]
    eY = [gamma_inv * (2.0 * q1q2),
          gamma_inv * (1.0 + q1s - q2s),
          gamma_inv * (2.0 * q2)]
    eZ = [gamma_inv * (2.0 * q1),
          gamma_inv * (-2.0 * q2),
          gamma_inv * (1.0 - q1s - q2s)]

    # --- Cartesian position from equinoctial (Eq. 41) ---
    x_cart = X * eX[0] + Y * eY[0]
    y_cart = X * eX[1] + Y * eY[1]
    z_cart = X * eX[2] + Y * eY[2]

    # --- Cartesian velocity (Eq. 43) ---
    # Need h first: h = sqrt(c^2 - 2*r^2*U)
    t_expr = hy.time - float(time_origin)
    U_val = perturbation.U_expr(x_cart, y_cart, z_cart, r, t_expr, pars)
    h = hy.sqrt(c * c - 2.0 * r * r * U_val)
    h_minus_c = -2.0 * r * r * U_val / (h + c)

    Xdot = rdot * cosL - (h / r) * sinL
    Ydot = rdot * sinL + (h / r) * cosL

    vx_cart = Xdot * eX[0] + Ydot * eY[0]
    vy_cart = Xdot * eX[1] + Ydot * eY[1]
    vz_cart = Xdot * eX[2] + Ydot * eY[2]

    # --- Force components ---
    # Conservative gradient of U
    dUdx, dUdy, dUdz = perturbation.grad_U_expr(
        x_cart, y_cart, z_cart, r, t_expr, pars
    )

    # Non-conservative acceleration P
    Px, Py, Pz = perturbation.P_expr(
        x_cart, y_cart, z_cart, vx_cart, vy_cart, vz_cart, r, t_expr, pars
    )

    # Total perturbation force F = P - grad(U) (Eq. 3)
    Fx = Px - dUdx
    Fy = Py - dUdy
    Fz = Pz - dUdz

    # Orbital frame unit vectors
    er = [eX[i] * cosL + eY[i] * sinL for i in range(3)]
    ef = [eY[i] * cosL - eX[i] * sinL for i in range(3)]
    # eh = eZ

    # Force projections
    F_h = sum(Fx_i * ez_i for Fx_i, ez_i in zip([Fx, Fy, Fz], eZ))

    # Non-conservative projections (for E_dot)
    P_r = sum(Px_i * er_i for Px_i, er_i in zip([Px, Py, Pz], er))
    P_f = sum(Px_i * ef_i for Px_i, ef_i in zip([Px, Py, Pz], ef))

    # --- Angular rates (Eq. 14, 52) ---
    w_X = (X / h) * F_h
    w_Y = (Y / h) * F_h
    w_h = w_X * q1 - w_Y * q2

    d = h_minus_c / (r * r)

    # --- 2U - rF_r (using Euler identity for conservative part) ---
    r_dot_gradU = x_cart * dUdx + y_cart * dUdy + z_cart * dUdz
    two_U_minus_rFr = 2.0 * U_val + r_dot_gradU - r * P_r

    # --- Energy derivative E_dot (Eq. 46) ---
    U_t = perturbation.U_t_expr(x_cart, y_cart, z_cart, r, t_expr, pars)
    E_dot = U_t + rdot * P_r + (h / r) * P_f

    # --- Generalized semi-latus rectum (Eq. 19) ---
    rho = c * c / mu

    # --- Assemble ODEs ---

    # nu_dot (Eq. 45)
    nu_dot = -3.0 * (nu / (mu * mu)) ** (1.0 / 3.0) * E_dot

    # p1_dot (Eq. 47)
    p1_dot = (p2 * (d - w_h)
              + (1.0 / c) * (X / a + 2.0 * p2) * two_U_minus_rFr
              + (1.0 / (c * c)) * (Y * (r + rho) + r * r * p1) * E_dot)

    # p2_dot (Eq. 48)
    p2_dot = (p1 * (w_h - d)
              - (1.0 / c) * (Y / a + 2.0 * p1) * two_U_minus_rFr
              + (1.0 / (c * c)) * (X * (r + rho) + r * r * p2) * E_dot)

    # L_dot (Eq. 49)
    L_dot = (nu + d - w_h
             + (1.0 / c) * (1.0 / alpha + alpha * (p1 * sinK + p2 * cosK))
               * two_U_minus_rFr
             + (rdot * alpha) / (mu * c) * (r + rho) * E_dot)

    # K_dot derived from L_dot: K = L - p1*cosK + p2*sinK
    # K_dot * (r/a) = L_dot - p1_dot*cosK + p2_dot*sinK
    K_dot = (a / r) * (L_dot - p1_dot * cosK + p2_dot * sinK)

    # q1_dot (Eq. 50): q1_dot = (gamma/2) * w_Y
    q1_dot = 0.5 * gamma * w_Y

    # q2_dot (Eq. 51): q2_dot = (gamma/2) * w_X
    q2_dot = 0.5 * gamma * w_X

    sys = [
        (nu, nu_dot), (p1, p1_dot), (p2, p2_dot),
        (K, K_dot), (q1, q1_dot), (q2, q2_dot),
    ]
    return sys, state_vars, par_map
