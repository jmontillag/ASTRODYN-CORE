"""Taylor series order-1 computation for J2-perturbed GEqOE propagator.

Extracts the setup and first-order Taylor expansion from the legacy
monolithic ``propagator.py`` (lines ~162-673).  All intermediate values
are stored in ``context.scratch`` so that higher-order modules can read
them.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.propagation.geqoe._math_compat import (
    derivatives_of_inverse,
    derivatives_of_inverse_wrt_param,
    derivatives_of_product,
    derivatives_of_product_wrt_param,
)
from astrodyn_core.propagation.geqoe.state import GEqOEPropagationContext
from astrodyn_core.propagation.geqoe.utils import solve_kep_gen


def compute_order_1(ctx: GEqOEPropagationContext) -> None:  # noqa: C901
    """Compute the order-1 Taylor expansion and populate *ctx*."""
    s = ctx.scratch  # shorthand
    dt_norm = ctx.dt_norm
    M = len(dt_norm)
    T = ctx.constants.time_scale
    mu_norm = ctx.constants.mu_norm  # 1.0
    A = ctx.constants.a_half_j2  # J2 / 2

    # --- Extract initial state ---
    st = ctx.initial_state
    nu_0 = st.nu * T
    q1_0, q2_0, p1_0, p2_0 = st.q1, st.q2, st.p1, st.p2
    Lr_0 = np.mod(st.lr, 2 * np.pi)

    # Store for later orders
    s["nu_0"] = nu_0
    s["q1_0"] = q1_0
    s["q2_0"] = q2_0
    s["p1_0"] = p1_0
    s["p2_0"] = p2_0
    s["Lr_0"] = Lr_0
    s["T"] = T
    s["mu_norm"] = mu_norm
    s["A"] = A

    # --- Initialize output arrays ---
    y_prop = np.zeros((M, 6))
    y_y0 = np.zeros((M, 6, 6))
    map_components = np.zeros((6, ctx.order))

    # --- Initial state calculations ---
    K_0 = solve_kep_gen(np.array([Lr_0]), np.array([p1_0]), np.array([p2_0]))[0]

    q1s, q2s = q1_0**2, q2_0**2
    p1s, p2s = p1_0**2, p2_0**2
    gs = p1s + p2s
    beta = np.sqrt(1 - gs)
    alpha = 1.0 / (1 + beta)
    sinK, cosK = np.sin(K_0), np.cos(K_0)

    a = (mu_norm / nu_0**2) ** (1 / 3)
    X = a * (alpha * p1_0 * p2_0 * sinK + (1 - alpha * p1s) * cosK - p2_0)
    Y = a * (alpha * p1_0 * p2_0 * cosK + (1 - alpha * p2s) * sinK - p1_0)

    r = a * (1 - p1_0 * sinK - p2_0 * cosK)
    rp = np.sqrt(mu_norm * a) / r * (p2_0 * sinK - p1_0 * cosK)

    cosL = X / r
    sinL = Y / r

    c = (mu_norm**2 / nu_0) ** (1 / 3) * beta
    zg = 2 * (Y * q2_0 - X * q1_0) / (r * (1 + q1s + q2s))

    U = -A / r**3 * (1 - 3 * zg**2)

    h = np.sqrt(c**2 - 2 * r**2 * U)
    Xp = rp * cosL - h / r * sinL
    Yp = rp * sinL + h / r * cosL

    r2 = r**2
    r3 = r**3

    r_vector = np.array([r, rp])
    fir, firp = derivatives_of_inverse(r_vector)

    r3p = 3 * r2 * rp
    r3_vector = np.array([r3, r3p])
    fir3, fir3p = derivatives_of_inverse(r3_vector)

    f2rp = derivatives_of_product(r_vector, True)

    r2_vector = np.array([r2, 2 * f2rp])
    fir2, fir2p = derivatives_of_inverse(r2_vector)

    # beta functions
    beta_vector = np.array([beta])
    fib = 1 / beta

    # c functions
    c_vector = np.array([c])
    fic = 1 / c

    # beta+1 functions
    bm1_vector = np.array([beta + 1])
    fibm1 = alpha
    alpha_vector = np.array([fibm1])
    fialpha = 1 / alpha
    h_vector = np.array([h])
    fih = 1 / h
    hr = h * fir

    # INTERMEDIATE COMPUTATIONS
    delta = (1 - q1s - q2s)
    hr3 = h * r3
    I = 3 * A / hr3 * zg * delta
    d = (h - c) / r2
    wh = I * zg
    xi1 = X / a + 2 * p2_0
    xi2 = Y / a + 2 * p1_0
    GAMMA_ = fialpha + alpha * (1 - r / a)

    # Eq of motion
    p1p_0 = p2_0 * (d - wh) - fic * xi1 * U
    p2p_0 = p1_0 * (wh - d) + fic * xi2 * U
    Lrp_0 = nu_0 + d - wh - fic * GAMMA_ * U
    q1p_0 = -I * sinL
    q2p_0 = -I * cosL

    y_prop[:, 0] = nu_0
    y_prop[:, 1] = q1_0 + q1p_0 * dt_norm
    y_prop[:, 2] = q2_0 + q2p_0 * dt_norm
    y_prop[:, 3] = p1_0 + p1p_0 * dt_norm
    y_prop[:, 4] = p2_0 + p2p_0 * dt_norm
    y_prop[:, 5] = Lr_0 + Lrp_0 * dt_norm

    map_components[:, 0] = [0, q1p_0, q2p_0, p1p_0, p2p_0, Lrp_0]

    # ------------------------------------------------------------------ #
    # DERIVATIVES WRT THE INITIAL CONDITIONS                              #
    # ------------------------------------------------------------------ #

    # a
    a_nu = mu_norm ** (1 / 3) * (-2 / 3) / nu_0 ** (5 / 3)

    # beta
    beta_nu, beta_q1, beta_q2, beta_Lr = 0, 0, 0, 0
    beta_p1 = -p1_0 * fib
    beta_p2 = -p2_0 * fib

    # c
    c_q1, c_q2, c_Lr = 0, 0, 0
    c_nu = -beta * mu_norm ** (2 / 3) / (3 * nu_0 ** (4 / 3))
    c_p1 = mu_norm ** (2 / 3) / nu_0 ** (1 / 3) * beta_p1
    c_p2 = mu_norm ** (2 / 3) / nu_0 ** (1 / 3) * beta_p2

    c_nu_vector = np.array([c_nu])
    c_p1_vector = np.array([c_p1])
    c_p2_vector = np.array([c_p2])
    fic_nu = derivatives_of_inverse_wrt_param(c_vector, c_nu_vector, True)
    fic_p1 = derivatives_of_inverse_wrt_param(c_vector, c_p1_vector, True)
    fic_p2 = derivatives_of_inverse_wrt_param(c_vector, c_p2_vector, True)
    fic_q1, fic_q2, fic_Lr = 0, 0, 0

    # K
    K_Lr = 1 / (1 - p1_0 * sinK - p2_0 * cosK)
    K_p1 = cosK / (-1 + p1_0 * sinK + p2_0 * cosK)
    K_p2 = sinK / (1 - p1_0 * sinK - p2_0 * cosK)

    # r
    r_q1 = 0
    r_q2 = 0
    r_nu = a_nu * (1 - p1_0 * sinK - p2_0 * cosK)
    r_Lr = a * (-p1_0 * cosK * K_Lr + p2_0 * sinK * K_Lr)
    r_p1 = a * (-sinK - p1_0 * cosK * K_p1 + p2_0 * sinK * K_p1)
    r_p2 = a * (-p1_0 * cosK * K_p2 - cosK + p2_0 * sinK * K_p2)

    # alpha
    alpha_nu = 0
    alpha_q1 = 0
    alpha_q2 = 0
    alpha_Lr = 0
    alpha_p1 = -beta_p1 / (1 + beta) ** 2
    alpha_p2 = -beta_p2 / (1 + beta) ** 2

    alpha_p1_vector = np.array([alpha_p1])
    alpha_p2_vector = np.array([alpha_p2])
    fialpha_p1 = derivatives_of_inverse_wrt_param(np.array([alpha]), alpha_p1_vector, True)
    fialpha_p2 = derivatives_of_inverse_wrt_param(np.array([alpha]), alpha_p2_vector, True)
    fialpha_nu = 0
    fialpha_q1 = 0
    fialpha_q2 = 0
    fialpha_Lr = 0

    # X
    X_q1 = 0
    X_q2 = 0
    X_nu = a_nu * (alpha * p1_0 * p2_0 * sinK + (1 - alpha * p1s) * cosK - p2_0)
    X_p1 = a * (
        alpha_p1 * p1_0 * p2_0 * sinK
        + alpha * (p2_0 * sinK + p1_0 * p2_0 * cosK * K_p1)
        - (alpha_p1 * p1s + alpha * 2 * p1_0) * cosK
        - (1 - alpha * p1s) * sinK * K_p1
    )
    X_p2 = a * (
        alpha_p2 * p1_0 * p2_0 * sinK
        + alpha * (p1_0 * sinK + p1_0 * p2_0 * cosK * K_p2)
        - alpha_p2 * p1s * cosK
        - (1 - alpha * p1s) * sinK * K_p2
        - 1
    )
    X_Lr = a * (alpha * p1_0 * p2_0 * cosK * K_Lr - (1 - alpha * p1s) * sinK * K_Lr)

    # Y
    Y_q1 = 0
    Y_q2 = 0
    Y_nu = a_nu * (alpha * p1_0 * p2_0 * cosK + (1 - alpha * p2s) * sinK - p1_0)
    Y_p1 = a * (
        alpha_p1 * p1_0 * p2_0 * cosK
        + alpha * (p2_0 * cosK - p1_0 * p2_0 * sinK * K_p1)
        - alpha_p1 * p2s * sinK
        + (1 - alpha * p2s) * cosK * K_p1
        - 1
    )
    Y_p2 = a * (
        alpha_p2 * p1_0 * p2_0 * cosK
        + alpha * (p1_0 * cosK - p1_0 * p2_0 * sinK * K_p2)
        - (alpha_p2 * p2_0**2 + alpha * 2 * p2_0) * sinK
        + (1 - alpha * p2s) * cosK * K_p2
    )
    Y_Lr = a * (-alpha * p1_0 * p2_0 * sinK * K_Lr + (1 - alpha * p2s) * cosK * K_Lr)

    # cosL sinL
    r_nu_vector = np.array([r_nu])
    r_Lr_vector = np.array([r_Lr])
    r_p1_vector = np.array([r_p1])
    r_p2_vector = np.array([r_p2])

    fir_nu = derivatives_of_inverse_wrt_param(np.array([r]), r_nu_vector, True)
    fir_Lr = derivatives_of_inverse_wrt_param(np.array([r]), r_Lr_vector, True)
    fir_p1 = derivatives_of_inverse_wrt_param(np.array([r]), r_p1_vector, True)
    fir_p2 = derivatives_of_inverse_wrt_param(np.array([r]), r_p2_vector, True)
    fir_q1 = 0
    fir_q2 = 0

    cosL_q1 = 0
    cosL_q2 = 0
    cosL_nu = X_nu * fir + X * fir_nu
    cosL_p1 = X_p1 * fir + X * fir_p1
    cosL_p2 = X_p2 * fir + X * fir_p2
    cosL_Lr = X_Lr * fir + X * fir_Lr

    sinL_q1 = 0
    sinL_q2 = 0
    sinL_nu = Y_nu * fir + Y * fir_nu
    sinL_p1 = Y_p1 * fir + Y * fir_p1
    sinL_p2 = Y_p2 * fir + Y * fir_p2
    sinL_Lr = Y_Lr * fir + Y * fir_Lr

    # zg
    C_nu = 2 * (Y_nu * q2_0 - X_nu * q1_0)
    C_p1 = 2 * (Y_p1 * q2_0 - X_p1 * q1_0)
    C_p2 = 2 * (Y_p2 * q2_0 - X_p2 * q1_0)
    C_Lr = 2 * (Y_Lr * q2_0 - X_Lr * q1_0)
    C_q1 = -2 * X
    C_q2 = 2 * Y

    qs = 1 + q1s + q2s

    D_nu = r_nu * qs
    D_p1 = r_p1 * qs
    D_p2 = r_p2 * qs
    D_Lr = r_Lr * qs
    D_q1 = r * 2 * q1_0
    D_q2 = r * 2 * q2_0

    C = 2 * (Y * q2_0 - X * q1_0)
    D = r * qs
    fiD = 1 / D
    Delta_vector = np.array([D])

    D_nu_vector = np.array([D_nu])
    D_Lr_vector = np.array([D_Lr])
    D_q1_vector = np.array([D_q1])
    D_q2_vector = np.array([D_q2])
    D_p1_vector = np.array([D_p1])
    D_p2_vector = np.array([D_p2])
    fiD_nu = derivatives_of_inverse_wrt_param(Delta_vector, D_nu_vector, True)
    fiD_Lr = derivatives_of_inverse_wrt_param(Delta_vector, D_Lr_vector, True)
    fiD_q1 = derivatives_of_inverse_wrt_param(Delta_vector, D_q1_vector, True)
    fiD_q2 = derivatives_of_inverse_wrt_param(Delta_vector, D_q2_vector, True)
    fiD_p1 = derivatives_of_inverse_wrt_param(Delta_vector, D_p1_vector, True)
    fiD_p2 = derivatives_of_inverse_wrt_param(Delta_vector, D_p2_vector, True)

    zg_nu = C_nu * fiD + C * fiD_nu
    zg_Lr = C_Lr * fiD + C * fiD_Lr
    zg_q1 = C_q1 * fiD + C * fiD_q1
    zg_q2 = C_q2 * fiD + C * fiD_q2
    zg_p1 = C_p1 * fiD + C * fiD_p1
    zg_p2 = C_p2 * fiD + C * fiD_p2

    # U
    fUz = 1 - 3 * zg**2
    fUz_nu = -6 * zg * zg_nu
    fUz_Lr = -6 * zg * zg_Lr
    fUz_q1 = -6 * zg * zg_q1
    fUz_q2 = -6 * zg * zg_q2
    fUz_p1 = -6 * zg * zg_p1
    fUz_p2 = -6 * zg * zg_p2

    # fir3 = 1/r3
    r3_nu = 3 * r2 * r_nu
    r3_Lr = 3 * r2 * r_Lr
    r3_q1 = 3 * r2 * r_q1
    r3_q2 = 3 * r2 * r_q2
    r3_p1 = 3 * r2 * r_p1
    r3_p2 = 3 * r2 * r_p2
    r3_nu_vector = np.array([r3_nu])
    r3_Lr_vector = np.array([r3_Lr])
    r3_q1_vector = np.array([r3_q1])
    r3_q2_vector = np.array([r3_q2])
    r3_p1_vector = np.array([r3_p1])
    r3_p2_vector = np.array([r3_p2])

    fir3_nu = derivatives_of_inverse_wrt_param(np.array([r3]), r3_nu_vector, True)
    fir3_Lr = derivatives_of_inverse_wrt_param(np.array([r3]), r3_Lr_vector, True)
    fir3_q1 = derivatives_of_inverse_wrt_param(np.array([r3]), r3_q1_vector, True)
    fir3_q2 = derivatives_of_inverse_wrt_param(np.array([r3]), r3_q2_vector, True)
    fir3_p1 = derivatives_of_inverse_wrt_param(np.array([r3]), r3_p1_vector, True)
    fir3_p2 = derivatives_of_inverse_wrt_param(np.array([r3]), r3_p2_vector, True)

    U_nu = -A * (fUz_nu * fir3 + fUz * fir3_nu)
    U_Lr = -A * (fUz_Lr * fir3 + fUz * fir3_Lr)
    U_q1 = -A * (fUz_q1 * fir3 + fUz * fir3_q1)
    U_q2 = -A * (fUz_q2 * fir3 + fUz * fir3_q2)
    U_p1 = -A * (fUz_p1 * fir3 + fUz * fir3_p1)
    U_p2 = -A * (fUz_p2 * fir3 + fUz * fir3_p2)

    # h
    h_nu = (c * c_nu - 2 * r * r_nu * U - r2 * U_nu) / h
    h_Lr = (c * c_Lr - 2 * r * r_Lr * U - r2 * U_Lr) / h
    h_q1 = (c * c_q1 - 2 * r * r_q1 * U - r2 * U_q1) / h
    h_q2 = (c * c_q2 - 2 * r * r_q2 * U - r2 * U_q2) / h
    h_p1 = (c * c_p1 - 2 * r * r_p1 * U - r2 * U_p1) / h
    h_p2 = (c * c_p2 - 2 * r * r_p2 * U - r2 * U_p2) / h

    h_nu_vector = np.array([h_nu])
    h_Lr_vector = np.array([h_Lr])
    h_q1_vector = np.array([h_q1])
    h_q2_vector = np.array([h_q2])
    h_p1_vector = np.array([h_p1])
    h_p2_vector = np.array([h_p2])

    fih_nu = derivatives_of_inverse_wrt_param(h_vector, h_nu_vector, True)
    fih_Lr = derivatives_of_inverse_wrt_param(h_vector, h_Lr_vector, True)
    fih_q1 = derivatives_of_inverse_wrt_param(h_vector, h_q1_vector, True)
    fih_q2 = derivatives_of_inverse_wrt_param(h_vector, h_q2_vector, True)
    fih_p1 = derivatives_of_inverse_wrt_param(h_vector, h_p1_vector, True)
    fih_p2 = derivatives_of_inverse_wrt_param(h_vector, h_p2_vector, True)

    # hr3
    hr3_nu = h_nu * r3 + h * r3_nu
    hr3_Lr = h_Lr * r3 + h * r3_Lr
    hr3_q1 = h_q1 * r3 + h * r3_q1
    hr3_q2 = h_q2 * r3 + h * r3_q2
    hr3_p1 = h_p1 * r3 + h * r3_p1
    hr3_p2 = h_p2 * r3 + h * r3_p2

    hr3_nu_vector = np.array([hr3_nu])
    hr3_Lr_vector = np.array([hr3_Lr])
    hr3_q1_vector = np.array([hr3_q1])
    hr3_q2_vector = np.array([hr3_q2])
    hr3_p1_vector = np.array([hr3_p1])
    hr3_p2_vector = np.array([hr3_p2])

    fihr3_nu = derivatives_of_inverse_wrt_param(np.array([hr3]), hr3_nu_vector, True)
    fihr3_Lr = derivatives_of_inverse_wrt_param(np.array([hr3]), hr3_Lr_vector, True)
    fihr3_q1 = derivatives_of_inverse_wrt_param(np.array([hr3]), hr3_q1_vector, True)
    fihr3_q2 = derivatives_of_inverse_wrt_param(np.array([hr3]), hr3_q2_vector, True)
    fihr3_p1 = derivatives_of_inverse_wrt_param(np.array([hr3]), hr3_p1_vector, True)
    fihr3_p2 = derivatives_of_inverse_wrt_param(np.array([hr3]), hr3_p2_vector, True)

    # delta
    delta_nu = 0
    delta_Lr = 0
    delta_p1 = 0
    delta_p2 = 0
    delta_q1 = -2 * q1_0
    delta_q2 = -2 * q2_0

    # I
    fihr3 = 1 / hr3
    I_nu = 3 * A * (zg_nu * delta * fihr3 + zg * delta * fihr3_nu)
    I_Lr = 3 * A * (zg_Lr * delta * fihr3 + zg * delta * fihr3_Lr)
    I_p1 = 3 * A * (zg_p1 * delta * fihr3 + zg * delta * fihr3_p1)
    I_p2 = 3 * A * (zg_p2 * delta * fihr3 + zg * delta * fihr3_p2)
    I_q1 = 3 * A * ((zg_q1 * delta + zg * delta_q1) * fihr3 + zg * delta * fihr3_q1)
    I_q2 = 3 * A * ((zg_q2 * delta + zg * delta_q2) * fihr3 + zg * delta * fihr3_q2)

    # r2
    r2_nu = 2 * r * r_nu
    r2_Lr = 2 * r * r_Lr
    r2_q1 = 2 * r * r_q1
    r2_q2 = 2 * r * r_q2
    r2_p1 = 2 * r * r_p1
    r2_p2 = 2 * r * r_p2

    r2_nu_vector = np.array([r2_nu])
    r2_Lr_vector = np.array([r2_Lr])
    r2_q1_vector = np.array([r2_q1])
    r2_q2_vector = np.array([r2_q2])
    r2_p1_vector = np.array([r2_p1])
    r2_p2_vector = np.array([r2_p2])

    fir2_nu = derivatives_of_inverse_wrt_param(np.array([r2]), r2_nu_vector, True)
    fir2_Lr = derivatives_of_inverse_wrt_param(np.array([r2]), r2_Lr_vector, True)
    fir2_q1 = derivatives_of_inverse_wrt_param(np.array([r2]), r2_q1_vector, True)
    fir2_q2 = derivatives_of_inverse_wrt_param(np.array([r2]), r2_q2_vector, True)
    fir2_p1 = derivatives_of_inverse_wrt_param(np.array([r2]), r2_p1_vector, True)
    fir2_p2 = derivatives_of_inverse_wrt_param(np.array([r2]), r2_p2_vector, True)

    # rp
    rpn = p2_0 * sinL - p1_0 * cosL

    rpn_nu = p2_0 * sinL_nu - p1_0 * cosL_nu
    rpn_p1 = p2_0 * sinL_p1 - cosL - p1_0 * cosL_p1
    rpn_p2 = sinL + p2_0 * sinL_p2 - p1_0 * cosL_p2
    rpn_Lr = p2_0 * sinL_Lr - p1_0 * cosL_Lr
    rpn_q1 = p2_0 * sinL_q1 - p1_0 * cosL_q1
    rpn_q2 = p2_0 * sinL_q2 - p1_0 * cosL_q2

    rp_nu = mu_norm * (rpn_nu * fic + rpn * fic_nu)
    rp_Lr = mu_norm * (rpn_Lr * fic + rpn * fic_Lr)
    rp_q1 = mu_norm * (rpn_q1 * fic + rpn * fic_q1)
    rp_q2 = mu_norm * (rpn_q2 * fic + rpn * fic_q2)
    rp_p1 = mu_norm * (rpn_p1 * fic + rpn * fic_p1)
    rp_p2 = mu_norm * (rpn_p2 * fic + rpn * fic_p2)

    r_nu_vector = np.array([r_nu, rp_nu])
    r_Lr_vector = np.array([r_Lr, rp_Lr])
    r_q1_vector = np.array([r_q1, rp_q1])
    r_q2_vector = np.array([r_q2, rp_q2])
    r_p1_vector = np.array([r_p1, rp_p1])
    r_p2_vector = np.array([r_p2, rp_p2])

    f2rp_nu = derivatives_of_product_wrt_param(r_vector, r_nu_vector, True)
    f2rp_Lr = derivatives_of_product_wrt_param(r_vector, r_Lr_vector, True)
    f2rp_q1 = derivatives_of_product_wrt_param(r_vector, r_q1_vector, True)
    f2rp_q2 = derivatives_of_product_wrt_param(r_vector, r_q2_vector, True)
    f2rp_p1 = derivatives_of_product_wrt_param(r_vector, r_p1_vector, True)
    f2rp_p2 = derivatives_of_product_wrt_param(r_vector, r_p2_vector, True)

    firp_nu = derivatives_of_inverse_wrt_param(r_vector, r_nu_vector, True)
    firp_Lr = derivatives_of_inverse_wrt_param(r_vector, r_Lr_vector, True)
    firp_q1 = derivatives_of_inverse_wrt_param(r_vector, r_q1_vector, True)
    firp_q2 = derivatives_of_inverse_wrt_param(r_vector, r_q2_vector, True)
    firp_p1 = derivatives_of_inverse_wrt_param(r_vector, r_p1_vector, True)
    firp_p2 = derivatives_of_inverse_wrt_param(r_vector, r_p2_vector, True)

    r2_nu_vector = np.array([r2_nu, 2 * f2rp_nu])
    r2_Lr_vector = np.array([r2_Lr, 2 * f2rp_Lr])
    r2_q1_vector = np.array([r2_q1, 2 * f2rp_q1])
    r2_q2_vector = np.array([r2_q2, 2 * f2rp_q2])
    r2_p1_vector = np.array([r2_p1, 2 * f2rp_p1])
    r2_p2_vector = np.array([r2_p2, 2 * f2rp_p2])

    fir2p_nu = derivatives_of_inverse_wrt_param(r2_vector, r2_nu_vector, True)
    fir2p_Lr = derivatives_of_inverse_wrt_param(r2_vector, r2_Lr_vector, True)
    fir2p_q1 = derivatives_of_inverse_wrt_param(r2_vector, r2_q1_vector, True)
    fir2p_q2 = derivatives_of_inverse_wrt_param(r2_vector, r2_q2_vector, True)
    fir2p_p1 = derivatives_of_inverse_wrt_param(r2_vector, r2_p1_vector, True)
    fir2p_p2 = derivatives_of_inverse_wrt_param(r2_vector, r2_p2_vector, True)

    # r3p = 3*r2*rp
    r3p_nu = 3 * (2 * r * r_nu * rp + r2 * rp_nu)
    r3p_Lr = 3 * (2 * r * r_Lr * rp + r2 * rp_Lr)
    r3p_q1 = 3 * (2 * r * r_q1 * rp + r2 * rp_q1)
    r3p_q2 = 3 * (2 * r * r_q2 * rp + r2 * rp_q2)
    r3p_p1 = 3 * (2 * r * r_p1 * rp + r2 * rp_p1)
    r3p_p2 = 3 * (2 * r * r_p2 * rp + r2 * rp_p2)

    r3_nu_vector = np.array([r3_nu, r3p_nu])
    r3_Lr_vector = np.array([r3_Lr, r3p_Lr])
    r3_q1_vector = np.array([r3_q1, r3p_q1])
    r3_q2_vector = np.array([r3_q2, r3p_q2])
    r3_p1_vector = np.array([r3_p1, r3p_p1])
    r3_p2_vector = np.array([r3_p2, r3p_p2])

    fir3_p_nu = derivatives_of_inverse_wrt_param(r3_vector, r3_nu_vector, True)
    fir3_p_Lr = derivatives_of_inverse_wrt_param(r3_vector, r3_Lr_vector, True)
    fir3_p_q1 = derivatives_of_inverse_wrt_param(r3_vector, r3_q1_vector, True)
    fir3_p_q2 = derivatives_of_inverse_wrt_param(r3_vector, r3_q2_vector, True)
    fir3_p_p1 = derivatives_of_inverse_wrt_param(r3_vector, r3_p1_vector, True)
    fir3_p_p2 = derivatives_of_inverse_wrt_param(r3_vector, r3_p2_vector, True)

    # d
    d_nu = (h_nu - c_nu) * fir2 + (h - c) * fir2_nu
    d_Lr = (h_Lr - c_Lr) * fir2 + (h - c) * fir2_Lr
    d_q1 = (h_q1 - c_q1) * fir2 + (h - c) * fir2_q1
    d_q2 = (h_q2 - c_q2) * fir2 + (h - c) * fir2_q2
    d_p1 = (h_p1 - c_p1) * fir2 + (h - c) * fir2_p1
    d_p2 = (h_p2 - c_p2) * fir2 + (h - c) * fir2_p2

    # wh
    wh_nu = I_nu * zg + I * zg_nu
    wh_Lr = I_Lr * zg + I * zg_Lr
    wh_q1 = I_q1 * zg + I * zg_q1
    wh_q2 = I_q2 * zg + I * zg_q2
    wh_p1 = I_p1 * zg + I * zg_p1
    wh_p2 = I_p2 * zg + I * zg_p2

    # GAMMA
    GAMMA_nu = fialpha_nu + alpha_nu * (1 - r / a) + alpha * (-r_nu / a + r / a**2 * a_nu)
    GAMMA_Lr = fialpha_Lr + alpha_Lr * (1 - r / a) - alpha * r_Lr / a
    GAMMA_q1 = fialpha_q1 + alpha_q1 * (1 - r / a) - alpha * r_q1 / a
    GAMMA_q2 = fialpha_q2 + alpha_q2 * (1 - r / a) - alpha * r_q2 / a
    GAMMA_p1 = fialpha_p1 + alpha_p1 * (1 - r / a) - alpha * r_p1 / a
    GAMMA_p2 = fialpha_p2 + alpha_p2 * (1 - r / a) - alpha * r_p2 / a

    # xi1 xi2
    xi1_nu = X_nu / a - X / a**2 * a_nu
    xi2_nu = Y_nu / a - Y / a**2 * a_nu
    xi1_Lr = X_Lr / a
    xi2_Lr = Y_Lr / a
    xi1_q1 = X_q1 / a
    xi2_q1 = Y_q1 / a
    xi1_q2 = X_q2 / a
    xi2_q2 = Y_q2 / a
    xi1_p1 = X_p1 / a
    xi2_p1 = Y_p1 / a + 2
    xi1_p2 = X_p2 / a + 2
    xi2_p2 = Y_p2 / a

    # q1p derivatives
    q1p_nu = -I_nu * sinL - I * sinL_nu
    q1p_Lr = -I_Lr * sinL - I * sinL_Lr
    q1p_q1 = -I_q1 * sinL - I * sinL_q1
    q1p_q2 = -I_q2 * sinL - I * sinL_q2
    q1p_p1 = -I_p1 * sinL - I * sinL_p1
    q1p_p2 = -I_p2 * sinL - I * sinL_p2

    # q2p derivatives
    q2p_nu = -I_nu * cosL - I * cosL_nu
    q2p_Lr = -I_Lr * cosL - I * cosL_Lr
    q2p_q1 = -I_q1 * cosL - I * cosL_q1
    q2p_q2 = -I_q2 * cosL - I * cosL_q2
    q2p_p1 = -I_p1 * cosL - I * cosL_p1
    q2p_p2 = -I_p2 * cosL - I * cosL_p2

    # p1p derivatives
    p1p_nu = p2_0 * (d_nu - wh_nu) - (fic_nu * xi1 + fic * xi1_nu) * U - fic * xi1 * U_nu
    p1p_Lr = p2_0 * (d_Lr - wh_Lr) - (fic_Lr * xi1 + fic * xi1_Lr) * U - fic * xi1 * U_Lr
    p1p_q1 = p2_0 * (d_q1 - wh_q1) - (fic_q1 * xi1 + fic * xi1_q1) * U - fic * xi1 * U_q1
    p1p_q2 = p2_0 * (d_q2 - wh_q2) - (fic_q2 * xi1 + fic * xi1_q2) * U - fic * xi1 * U_q2
    p1p_p1 = p2_0 * (d_p1 - wh_p1) - (fic_p1 * xi1 + fic * xi1_p1) * U - fic * xi1 * U_p1
    p1p_p2 = (d - wh) + p2_0 * (d_p2 - wh_p2) - (fic_p2 * xi1 + fic * xi1_p2) * U - fic * xi1 * U_p2

    # p2p derivatives
    p2p_nu = p1_0 * (-d_nu + wh_nu) + (fic_nu * xi2 + fic * xi2_nu) * U + fic * xi2 * U_nu
    p2p_Lr = p1_0 * (-d_Lr + wh_Lr) + (fic_Lr * xi2 + fic * xi2_Lr) * U + fic * xi2 * U_Lr
    p2p_q1 = p1_0 * (-d_q1 + wh_q1) + (fic_q1 * xi2 + fic * xi2_q1) * U + fic * xi2 * U_q1
    p2p_q2 = p1_0 * (-d_q2 + wh_q2) + (fic_q2 * xi2 + fic * xi2_q2) * U + fic * xi2 * U_q2
    p2p_p1 = (-d + wh) + p1_0 * (-d_p1 + wh_p1) + (fic_p1 * xi2 + fic * xi2_p1) * U + fic * xi2 * U_p1
    p2p_p2 = p1_0 * (-d_p2 + wh_p2) + (fic_p2 * xi2 + fic * xi2_p2) * U + fic * xi2 * U_p2

    # Lrp derivatives
    Lrp_nu = 1 + d_nu - wh_nu - (fic_nu * GAMMA_ + fic * GAMMA_nu) * U - fic * GAMMA_ * U_nu
    Lrp_Lr = d_Lr - wh_Lr - (fic_Lr * GAMMA_ + fic * GAMMA_Lr) * U - fic * GAMMA_ * U_Lr
    Lrp_q1 = d_q1 - wh_q1 - (fic_q1 * GAMMA_ + fic * GAMMA_q1) * U - fic * GAMMA_ * U_q1
    Lrp_q2 = d_q2 - wh_q2 - (fic_q2 * GAMMA_ + fic * GAMMA_q2) * U - fic * GAMMA_ * U_q2
    Lrp_p1 = d_p1 - wh_p1 - (fic_p1 * GAMMA_ + fic * GAMMA_p1) * U - fic * GAMMA_ * U_p1
    Lrp_p2 = d_p2 - wh_p2 - (fic_p2 * GAMMA_ + fic * GAMMA_p2) * U - fic * GAMMA_ * U_p2

    # nu derivatives
    nu_nu = 1  # nu_Lr = 0; nu_q1 = 0; nu_q2 = 0 nu_p1 = 0 nu_p2 = 0
    # Lr derivatives
    Lr_nu = Lrp_nu * dt_norm
    Lr_Lr = 1 + Lrp_Lr * dt_norm
    Lr_q1 = Lrp_q1 * dt_norm
    Lr_q2 = Lrp_q2 * dt_norm
    Lr_p1 = Lrp_p1 * dt_norm
    Lr_p2 = Lrp_p2 * dt_norm
    # q1 derivatives
    q1_nu = q1p_nu * dt_norm
    q1_Lr = q1p_Lr * dt_norm
    q1_q1 = 1 + q1p_q1 * dt_norm
    q1_q2 = q1p_q2 * dt_norm
    q1_p1 = q1p_p1 * dt_norm
    q1_p2 = q1p_p2 * dt_norm
    # q2 derivatives
    q2_nu = q2p_nu * dt_norm
    q2_Lr = q2p_Lr * dt_norm
    q2_q1 = q2p_q1 * dt_norm
    q2_q2 = 1 + q2p_q2 * dt_norm
    q2_p1 = q2p_p1 * dt_norm
    q2_p2 = q2p_p2 * dt_norm
    # p1 derivatives
    p1_nu = p1p_nu * dt_norm
    p1_Lr = p1p_Lr * dt_norm
    p1_q1 = p1p_q1 * dt_norm
    p1_q2 = p1p_q2 * dt_norm
    p1_p1 = 1 + p1p_p1 * dt_norm
    p1_p2 = p1p_p2 * dt_norm
    # p2 derivatives
    p2_nu = p2p_nu * dt_norm
    p2_Lr = p2p_Lr * dt_norm
    p2_q1 = p2p_q1 * dt_norm
    p2_q2 = p2p_q2 * dt_norm
    p2_p1 = p2p_p1 * dt_norm
    p2_p2 = 1 + p2p_p2 * dt_norm

    # ------------------------------------------------------------------ #
    # Store everything in scratch for higher-order modules                #
    # ------------------------------------------------------------------ #
    # Scalars used by order 2+
    s["a"] = a;  s["a_nu"] = a_nu
    s["r"] = r;  s["rp"] = rp;  s["r2"] = r2;  s["r3"] = r3
    s["X"] = X;  s["Y"] = Y;  s["Xp"] = Xp;  s["Yp"] = Yp
    s["cosL"] = cosL;  s["sinL"] = sinL
    s["alpha"] = alpha;  s["beta"] = beta;  s["fib"] = fib;  s["fic"] = fic
    s["fih"] = fih;  s["fir"] = fir;  s["firp"] = firp;  s["fir2"] = fir2;  s["fir2p"] = fir2p
    s["fir3"] = fir3;  s["fir3p"] = fir3p
    s["fihr3"] = fihr3;  s["fiD"] = fiD;  s["fiDp"] = None  # set by order 2
    s["c"] = c;  s["h"] = h;  s["hr"] = hr;  s["hr3"] = hr3
    s["d"] = d;  s["wh"] = wh;  s["I"] = I;  s["U"] = U
    s["delta"] = delta;  s["zg"] = zg
    s["GAMMA_"] = GAMMA_
    s["xi1"] = xi1;  s["xi2"] = xi2
    s["fUz"] = fUz
    s["qs"] = qs;  s["q1s"] = q1s;  s["q2s"] = q2s;  s["p1s"] = p1s;  s["p2s"] = p2s
    s["C"] = C;  s["D"] = D
    s["rpn"] = rpn
    s["fialpha"] = fialpha
    s["f2rp"] = f2rp

    # EOM values
    s["p1p_0"] = p1p_0;  s["p2p_0"] = p2p_0;  s["Lrp_0"] = Lrp_0
    s["q1p_0"] = q1p_0;  s["q2p_0"] = q2p_0

    # Vectors (will be appended by higher orders)
    s["r_vector"] = r_vector;  s["r2_vector"] = r2_vector;  s["r3_vector"] = r3_vector
    s["beta_vector"] = beta_vector;  s["c_vector"] = c_vector
    s["h_vector"] = h_vector;  s["bm1_vector"] = bm1_vector
    s["alpha_vector"] = alpha_vector;  s["Delta_vector"] = Delta_vector

    # Partials of ALL intermediate quantities w.r.t. initial conditions
    # (naming: <var>_<param> where param in {nu, q1, q2, p1, p2, Lr})
    # r partials
    s["r_nu"] = r_nu;  s["r_Lr"] = r_Lr;  s["r_q1"] = r_q1;  s["r_q2"] = r_q2
    s["r_p1"] = r_p1;  s["r_p2"] = r_p2
    s["rp_nu"] = rp_nu;  s["rp_Lr"] = rp_Lr;  s["rp_q1"] = rp_q1;  s["rp_q2"] = rp_q2
    s["rp_p1"] = rp_p1;  s["rp_p2"] = rp_p2
    s["r2_nu"] = r2_nu;  s["r2_Lr"] = r2_Lr;  s["r2_q1"] = r2_q1;  s["r2_q2"] = r2_q2
    s["r2_p1"] = r2_p1;  s["r2_p2"] = r2_p2
    s["r3_nu"] = r3_nu;  s["r3_Lr"] = r3_Lr;  s["r3_q1"] = r3_q1;  s["r3_q2"] = r3_q2
    s["r3_p1"] = r3_p1;  s["r3_p2"] = r3_p2

    # beta/alpha partials
    s["beta_nu"] = beta_nu;  s["beta_Lr"] = beta_Lr;  s["beta_q1"] = beta_q1
    s["beta_q2"] = beta_q2;  s["beta_p1"] = beta_p1;  s["beta_p2"] = beta_p2
    s["alpha_nu"] = alpha_nu;  s["alpha_Lr"] = alpha_Lr;  s["alpha_q1"] = alpha_q1
    s["alpha_q2"] = alpha_q2;  s["alpha_p1"] = alpha_p1;  s["alpha_p2"] = alpha_p2
    s["fialpha_nu"] = fialpha_nu;  s["fialpha_Lr"] = fialpha_Lr
    s["fialpha_q1"] = fialpha_q1;  s["fialpha_q2"] = fialpha_q2
    s["fialpha_p1"] = fialpha_p1;  s["fialpha_p2"] = fialpha_p2

    # c partials
    s["c_nu"] = c_nu;  s["c_Lr"] = c_Lr;  s["c_q1"] = c_q1;  s["c_q2"] = c_q2
    s["c_p1"] = c_p1;  s["c_p2"] = c_p2
    s["fic_nu"] = fic_nu;  s["fic_Lr"] = fic_Lr;  s["fic_q1"] = fic_q1
    s["fic_q2"] = fic_q2;  s["fic_p1"] = fic_p1;  s["fic_p2"] = fic_p2

    # h partials
    s["h_nu"] = h_nu;  s["h_Lr"] = h_Lr;  s["h_q1"] = h_q1;  s["h_q2"] = h_q2
    s["h_p1"] = h_p1;  s["h_p2"] = h_p2
    s["fih_nu"] = fih_nu;  s["fih_Lr"] = fih_Lr;  s["fih_q1"] = fih_q1
    s["fih_q2"] = fih_q2;  s["fih_p1"] = fih_p1;  s["fih_p2"] = fih_p2

    # X, Y partials
    s["X_nu"] = X_nu;  s["X_Lr"] = X_Lr;  s["X_q1"] = X_q1;  s["X_q2"] = X_q2
    s["X_p1"] = X_p1;  s["X_p2"] = X_p2
    s["Y_nu"] = Y_nu;  s["Y_Lr"] = Y_Lr;  s["Y_q1"] = Y_q1;  s["Y_q2"] = Y_q2
    s["Y_p1"] = Y_p1;  s["Y_p2"] = Y_p2

    # cosL, sinL partials
    s["cosL_nu"] = cosL_nu;  s["cosL_Lr"] = cosL_Lr;  s["cosL_q1"] = cosL_q1
    s["cosL_q2"] = cosL_q2;  s["cosL_p1"] = cosL_p1;  s["cosL_p2"] = cosL_p2
    s["sinL_nu"] = sinL_nu;  s["sinL_Lr"] = sinL_Lr;  s["sinL_q1"] = sinL_q1
    s["sinL_q2"] = sinL_q2;  s["sinL_p1"] = sinL_p1;  s["sinL_p2"] = sinL_p2

    # zg partials
    s["zg_nu"] = zg_nu;  s["zg_Lr"] = zg_Lr;  s["zg_q1"] = zg_q1
    s["zg_q2"] = zg_q2;  s["zg_p1"] = zg_p1;  s["zg_p2"] = zg_p2

    # fUz, U partials
    s["fUz_nu"] = fUz_nu;  s["fUz_Lr"] = fUz_Lr;  s["fUz_q1"] = fUz_q1
    s["fUz_q2"] = fUz_q2;  s["fUz_p1"] = fUz_p1;  s["fUz_p2"] = fUz_p2
    s["U_nu"] = U_nu;  s["U_Lr"] = U_Lr;  s["U_q1"] = U_q1
    s["U_q2"] = U_q2;  s["U_p1"] = U_p1;  s["U_p2"] = U_p2

    # hr3 partials
    s["hr3_nu"] = hr3_nu;  s["hr3_Lr"] = hr3_Lr;  s["hr3_q1"] = hr3_q1
    s["hr3_q2"] = hr3_q2;  s["hr3_p1"] = hr3_p1;  s["hr3_p2"] = hr3_p2
    s["fihr3_nu"] = fihr3_nu;  s["fihr3_Lr"] = fihr3_Lr
    s["fihr3_q1"] = fihr3_q1;  s["fihr3_q2"] = fihr3_q2
    s["fihr3_p1"] = fihr3_p1;  s["fihr3_p2"] = fihr3_p2

    # delta partials
    s["delta_nu"] = delta_nu;  s["delta_Lr"] = delta_Lr
    s["delta_q1"] = delta_q1;  s["delta_q2"] = delta_q2
    s["delta_p1"] = delta_p1;  s["delta_p2"] = delta_p2

    # I partials
    s["I_nu"] = I_nu;  s["I_Lr"] = I_Lr;  s["I_q1"] = I_q1
    s["I_q2"] = I_q2;  s["I_p1"] = I_p1;  s["I_p2"] = I_p2

    # d, wh, GAMMA partials
    s["d_nu"] = d_nu;  s["d_Lr"] = d_Lr;  s["d_q1"] = d_q1
    s["d_q2"] = d_q2;  s["d_p1"] = d_p1;  s["d_p2"] = d_p2
    s["wh_nu"] = wh_nu;  s["wh_Lr"] = wh_Lr;  s["wh_q1"] = wh_q1
    s["wh_q2"] = wh_q2;  s["wh_p1"] = wh_p1;  s["wh_p2"] = wh_p2
    s["GAMMA_nu"] = GAMMA_nu;  s["GAMMA_Lr"] = GAMMA_Lr
    s["GAMMA_q1"] = GAMMA_q1;  s["GAMMA_q2"] = GAMMA_q2
    s["GAMMA_p1"] = GAMMA_p1;  s["GAMMA_p2"] = GAMMA_p2

    # xi partials
    s["xi1_nu"] = xi1_nu;  s["xi1_Lr"] = xi1_Lr;  s["xi1_q1"] = xi1_q1
    s["xi1_q2"] = xi1_q2;  s["xi1_p1"] = xi1_p1;  s["xi1_p2"] = xi1_p2
    s["xi2_nu"] = xi2_nu;  s["xi2_Lr"] = xi2_Lr;  s["xi2_q1"] = xi2_q1
    s["xi2_q2"] = xi2_q2;  s["xi2_p1"] = xi2_p1;  s["xi2_p2"] = xi2_p2

    # EOM partials
    s["q1p_nu"] = q1p_nu;  s["q1p_Lr"] = q1p_Lr;  s["q1p_q1"] = q1p_q1
    s["q1p_q2"] = q1p_q2;  s["q1p_p1"] = q1p_p1;  s["q1p_p2"] = q1p_p2
    s["q2p_nu"] = q2p_nu;  s["q2p_Lr"] = q2p_Lr;  s["q2p_q1"] = q2p_q1
    s["q2p_q2"] = q2p_q2;  s["q2p_p1"] = q2p_p1;  s["q2p_p2"] = q2p_p2
    s["p1p_nu"] = p1p_nu;  s["p1p_Lr"] = p1p_Lr;  s["p1p_q1"] = p1p_q1
    s["p1p_q2"] = p1p_q2;  s["p1p_p1"] = p1p_p1;  s["p1p_p2"] = p1p_p2
    s["p2p_nu"] = p2p_nu;  s["p2p_Lr"] = p2p_Lr;  s["p2p_q1"] = p2p_q1
    s["p2p_q2"] = p2p_q2;  s["p2p_p1"] = p2p_p1;  s["p2p_p2"] = p2p_p2
    s["Lrp_nu"] = Lrp_nu;  s["Lrp_Lr"] = Lrp_Lr;  s["Lrp_q1"] = Lrp_q1
    s["Lrp_q2"] = Lrp_q2;  s["Lrp_p1"] = Lrp_p1;  s["Lrp_p2"] = Lrp_p2

    # rpn partials
    s["rpn_nu"] = rpn_nu;  s["rpn_Lr"] = rpn_Lr;  s["rpn_q1"] = rpn_q1
    s["rpn_q2"] = rpn_q2;  s["rpn_p1"] = rpn_p1;  s["rpn_p2"] = rpn_p2

    # Inverse-derivative vectors for param partials
    s["r_nu_vector"] = r_nu_vector;  s["r_Lr_vector"] = r_Lr_vector
    s["r_q1_vector"] = r_q1_vector;  s["r_q2_vector"] = r_q2_vector
    s["r_p1_vector"] = r_p1_vector;  s["r_p2_vector"] = r_p2_vector
    s["r2_nu_vector"] = r2_nu_vector;  s["r2_Lr_vector"] = r2_Lr_vector
    s["r2_q1_vector"] = r2_q1_vector;  s["r2_q2_vector"] = r2_q2_vector
    s["r2_p1_vector"] = r2_p1_vector;  s["r2_p2_vector"] = r2_p2_vector
    s["r3_nu_vector"] = r3_nu_vector;  s["r3_Lr_vector"] = r3_Lr_vector
    s["r3_q1_vector"] = r3_q1_vector;  s["r3_q2_vector"] = r3_q2_vector
    s["r3_p1_vector"] = r3_p1_vector;  s["r3_p2_vector"] = r3_p2_vector
    s["D_nu"] = D_nu;  s["D_Lr"] = D_Lr;  s["D_q1"] = D_q1
    s["D_q2"] = D_q2;  s["D_p1"] = D_p1;  s["D_p2"] = D_p2
    s["fiD_nu"] = fiD_nu;  s["fiD_Lr"] = fiD_Lr;  s["fiD_q1"] = fiD_q1
    s["fiD_q2"] = fiD_q2;  s["fiD_p1"] = fiD_p1;  s["fiD_p2"] = fiD_p2
    s["C_nu"] = C_nu;  s["C_Lr"] = C_Lr;  s["C_q1"] = C_q1
    s["C_q2"] = C_q2;  s["C_p1"] = C_p1;  s["C_p2"] = C_p2

    # Higher-order inverse derivative values
    s["fir3_p_nu"] = fir3_p_nu;  s["fir3_p_Lr"] = fir3_p_Lr
    s["fir3_p_q1"] = fir3_p_q1;  s["fir3_p_q2"] = fir3_p_q2
    s["fir3_p_p1"] = fir3_p_p1;  s["fir3_p_p2"] = fir3_p_p2

    s["fir_nu"] = fir_nu;  s["fir_Lr"] = fir_Lr;  s["fir_q1"] = fir_q1
    s["fir_q2"] = fir_q2;  s["fir_p1"] = fir_p1;  s["fir_p2"] = fir_p2
    s["firp_nu"] = firp_nu;  s["firp_Lr"] = firp_Lr;  s["firp_q1"] = firp_q1
    s["firp_q2"] = firp_q2;  s["firp_p1"] = firp_p1;  s["firp_p2"] = firp_p2
    s["fir2_nu"] = fir2_nu;  s["fir2_Lr"] = fir2_Lr;  s["fir2_q1"] = fir2_q1
    s["fir2_q2"] = fir2_q2;  s["fir2_p1"] = fir2_p1;  s["fir2_p2"] = fir2_p2
    s["fir2p_nu"] = fir2p_nu;  s["fir2p_Lr"] = fir2p_Lr;  s["fir2p_q1"] = fir2p_q1
    s["fir2p_q2"] = fir2p_q2;  s["fir2p_p1"] = fir2p_p1;  s["fir2p_p2"] = fir2p_p2
    s["fir3_nu"] = fir3_nu;  s["fir3_Lr"] = fir3_Lr;  s["fir3_q1"] = fir3_q1
    s["fir3_q2"] = fir3_q2;  s["fir3_p1"] = fir3_p1;  s["fir3_p2"] = fir3_p2

    s["f2rp_nu"] = f2rp_nu;  s["f2rp_Lr"] = f2rp_Lr;  s["f2rp_q1"] = f2rp_q1
    s["f2rp_q2"] = f2rp_q2;  s["f2rp_p1"] = f2rp_p1;  s["f2rp_p2"] = f2rp_p2

    # STM partial accumulators (arrays of shape (M,) for each element)
    s["nu_nu"] = nu_nu
    s["Lr_nu"] = Lr_nu;  s["Lr_Lr"] = Lr_Lr;  s["Lr_q1"] = Lr_q1
    s["Lr_q2"] = Lr_q2;  s["Lr_p1"] = Lr_p1;  s["Lr_p2"] = Lr_p2
    s["q1_nu"] = q1_nu;  s["q1_Lr"] = q1_Lr;  s["q1_q1"] = q1_q1
    s["q1_q2"] = q1_q2;  s["q1_p1"] = q1_p1;  s["q1_p2"] = q1_p2
    s["q2_nu"] = q2_nu;  s["q2_Lr"] = q2_Lr;  s["q2_q1"] = q2_q1
    s["q2_q2"] = q2_q2;  s["q2_p1"] = q2_p1;  s["q2_p2"] = q2_p2
    s["p1_nu"] = p1_nu;  s["p1_Lr"] = p1_Lr;  s["p1_q1"] = p1_q1
    s["p1_q2"] = p1_q2;  s["p1_p1"] = p1_p1;  s["p1_p2"] = p1_p2
    s["p2_nu"] = p2_nu;  s["p2_Lr"] = p2_Lr;  s["p2_q1"] = p2_q1
    s["p2_q2"] = p2_q2;  s["p2_p1"] = p2_p1;  s["p2_p2"] = p2_p2

    # Store outputs on context
    ctx.y_prop = y_prop
    ctx.y_y0 = y_y0  # placeholder, filled at final assembly
    ctx.map_components = map_components
