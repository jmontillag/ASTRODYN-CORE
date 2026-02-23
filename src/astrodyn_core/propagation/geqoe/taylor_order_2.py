"""Taylor series order-2 computation for J2-perturbed GEqOE propagator.

Functions
---------
compute_coefficients_2 : dt-independent coefficient computation
evaluate_order_2       : dt-dependent polynomial evaluation + STM accumulation
compute_order_2        : thin wrapper calling both (preserves existing API)
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.propagation.geqoe._derivatives import (
    derivatives_of_inverse,
    derivatives_of_inverse_wrt_param,
    derivatives_of_product,
    derivatives_of_product_wrt_param,
)
from astrodyn_core.propagation.geqoe.state import GEqOEPropagationContext


def compute_coefficients_2(ctx: GEqOEPropagationContext) -> None:  # noqa: C901
    """Compute order-2 Taylor coefficients (dt-independent).

    Calls :func:`compute_coefficients_1` first, then computes all
    second-order intermediate values, EOM coefficients, and their
    partials w.r.t. initial conditions.  Stores everything into
    ``ctx.scratch`` for later orders.  Does **not** touch ``ctx.dt_norm``
    or the STM accumulators (those are dt-dependent).
    """
    from astrodyn_core.propagation.geqoe.taylor_order_1 import compute_coefficients_1
    compute_coefficients_1(ctx)

    s = ctx.scratch

    # --- Extract ALL needed values from scratch ---
    nu_0 = s["nu_0"]
    q1_0 = s["q1_0"]; q2_0 = s["q2_0"]
    p1_0 = s["p1_0"]; p2_0 = s["p2_0"]
    mu_norm = s["mu_norm"]
    A = s["A"]

    a = s["a"]; a_nu = s["a_nu"]
    r = s["r"]; rp = s["rp"]; r2 = s["r2"]; r3 = s["r3"]
    X = s["X"]; Y = s["Y"]; Xp = s["Xp"]; Yp = s["Yp"]
    cosL = s["cosL"]; sinL = s["sinL"]
    alpha = s["alpha"]; beta = s["beta"]
    fib = s["fib"]; fic = s["fic"]
    fih = s["fih"]; fir = s["fir"]; firp = s["firp"]
    fir2 = s["fir2"]; fir2p = s["fir2p"]
    fir3 = s["fir3"]; fir3p = s["fir3p"]
    fihr3 = s["fihr3"]; fiD = s["fiD"]
    c = s["c"]; h = s["h"]; hr = s["hr"]; hr3 = s["hr3"]
    d = s["d"]; wh = s["wh"]; I = s["I"]; U = s["U"]
    delta = s["delta"]; zg = s["zg"]
    GAMMA_ = s["GAMMA_"]
    xi1 = s["xi1"]; xi2 = s["xi2"]
    fUz = s["fUz"]
    qs = s["qs"]
    C = s["C"]; D = s["D"]
    rpn = s["rpn"]
    fialpha = s["fialpha"]
    f2rp = s["f2rp"]

    # EOM values
    p1p_0 = s["p1p_0"]; p2p_0 = s["p2p_0"]; Lrp_0 = s["Lrp_0"]
    q1p_0 = s["q1p_0"]; q2p_0 = s["q2p_0"]

    # Vectors (will be appended)
    r_vector = s["r_vector"]; r2_vector = s["r2_vector"]; r3_vector = s["r3_vector"]
    beta_vector = s["beta_vector"]; c_vector = s["c_vector"]
    h_vector = s["h_vector"]; bm1_vector = s["bm1_vector"]
    alpha_vector = s["alpha_vector"]; Delta_vector = s["Delta_vector"]

    # r partials
    r_nu = s["r_nu"]; r_Lr = s["r_Lr"]; r_q1 = s["r_q1"]; r_q2 = s["r_q2"]
    r_p1 = s["r_p1"]; r_p2 = s["r_p2"]
    rp_nu = s["rp_nu"]; rp_Lr = s["rp_Lr"]; rp_q1 = s["rp_q1"]; rp_q2 = s["rp_q2"]
    rp_p1 = s["rp_p1"]; rp_p2 = s["rp_p2"]
    r2_nu = s["r2_nu"]; r2_Lr = s["r2_Lr"]; r2_q1 = s["r2_q1"]; r2_q2 = s["r2_q2"]
    r2_p1 = s["r2_p1"]; r2_p2 = s["r2_p2"]
    r3_nu = s["r3_nu"]; r3_Lr = s["r3_Lr"]; r3_q1 = s["r3_q1"]; r3_q2 = s["r3_q2"]
    r3_p1 = s["r3_p1"]; r3_p2 = s["r3_p2"]

    # beta/alpha partials
    beta_nu = s["beta_nu"]; beta_Lr = s["beta_Lr"]; beta_q1 = s["beta_q1"]
    beta_q2 = s["beta_q2"]; beta_p1 = s["beta_p1"]; beta_p2 = s["beta_p2"]
    alpha_nu = s["alpha_nu"]; alpha_Lr = s["alpha_Lr"]; alpha_q1 = s["alpha_q1"]
    alpha_q2 = s["alpha_q2"]; alpha_p1 = s["alpha_p1"]; alpha_p2 = s["alpha_p2"]
    fialpha_nu = s["fialpha_nu"]; fialpha_Lr = s["fialpha_Lr"]
    fialpha_q1 = s["fialpha_q1"]; fialpha_q2 = s["fialpha_q2"]
    fialpha_p1 = s["fialpha_p1"]; fialpha_p2 = s["fialpha_p2"]

    # c partials
    c_nu = s["c_nu"]; c_Lr = s["c_Lr"]; c_q1 = s["c_q1"]; c_q2 = s["c_q2"]
    c_p1 = s["c_p1"]; c_p2 = s["c_p2"]
    fic_nu = s["fic_nu"]; fic_Lr = s["fic_Lr"]; fic_q1 = s["fic_q1"]
    fic_q2 = s["fic_q2"]; fic_p1 = s["fic_p1"]; fic_p2 = s["fic_p2"]

    # h partials
    h_nu = s["h_nu"]; h_Lr = s["h_Lr"]; h_q1 = s["h_q1"]; h_q2 = s["h_q2"]
    h_p1 = s["h_p1"]; h_p2 = s["h_p2"]
    fih_nu = s["fih_nu"]; fih_Lr = s["fih_Lr"]; fih_q1 = s["fih_q1"]
    fih_q2 = s["fih_q2"]; fih_p1 = s["fih_p1"]; fih_p2 = s["fih_p2"]

    # X, Y partials
    X_nu = s["X_nu"]; X_Lr = s["X_Lr"]; X_q1 = s["X_q1"]; X_q2 = s["X_q2"]
    X_p1 = s["X_p1"]; X_p2 = s["X_p2"]
    Y_nu = s["Y_nu"]; Y_Lr = s["Y_Lr"]; Y_q1 = s["Y_q1"]; Y_q2 = s["Y_q2"]
    Y_p1 = s["Y_p1"]; Y_p2 = s["Y_p2"]

    # cosL, sinL partials
    cosL_nu = s["cosL_nu"]; cosL_Lr = s["cosL_Lr"]; cosL_q1 = s["cosL_q1"]
    cosL_q2 = s["cosL_q2"]; cosL_p1 = s["cosL_p1"]; cosL_p2 = s["cosL_p2"]
    sinL_nu = s["sinL_nu"]; sinL_Lr = s["sinL_Lr"]; sinL_q1 = s["sinL_q1"]
    sinL_q2 = s["sinL_q2"]; sinL_p1 = s["sinL_p1"]; sinL_p2 = s["sinL_p2"]

    # zg partials
    zg_nu = s["zg_nu"]; zg_Lr = s["zg_Lr"]; zg_q1 = s["zg_q1"]
    zg_q2 = s["zg_q2"]; zg_p1 = s["zg_p1"]; zg_p2 = s["zg_p2"]

    # fUz, U partials
    fUz_nu = s["fUz_nu"]; fUz_Lr = s["fUz_Lr"]; fUz_q1 = s["fUz_q1"]
    fUz_q2 = s["fUz_q2"]; fUz_p1 = s["fUz_p1"]; fUz_p2 = s["fUz_p2"]
    U_nu = s["U_nu"]; U_Lr = s["U_Lr"]; U_q1 = s["U_q1"]
    U_q2 = s["U_q2"]; U_p1 = s["U_p1"]; U_p2 = s["U_p2"]

    # hr3 partials
    hr3_nu = s["hr3_nu"]; hr3_Lr = s["hr3_Lr"]; hr3_q1 = s["hr3_q1"]
    hr3_q2 = s["hr3_q2"]; hr3_p1 = s["hr3_p1"]; hr3_p2 = s["hr3_p2"]
    fihr3_nu = s["fihr3_nu"]; fihr3_Lr = s["fihr3_Lr"]
    fihr3_q1 = s["fihr3_q1"]; fihr3_q2 = s["fihr3_q2"]
    fihr3_p1 = s["fihr3_p1"]; fihr3_p2 = s["fihr3_p2"]

    # delta partials
    delta_nu = s["delta_nu"]; delta_Lr = s["delta_Lr"]
    delta_q1 = s["delta_q1"]; delta_q2 = s["delta_q2"]
    delta_p1 = s["delta_p1"]; delta_p2 = s["delta_p2"]

    # I partials
    I_nu = s["I_nu"]; I_Lr = s["I_Lr"]; I_q1 = s["I_q1"]
    I_q2 = s["I_q2"]; I_p1 = s["I_p1"]; I_p2 = s["I_p2"]

    # d, wh, GAMMA partials
    d_nu = s["d_nu"]; d_Lr = s["d_Lr"]; d_q1 = s["d_q1"]
    d_q2 = s["d_q2"]; d_p1 = s["d_p1"]; d_p2 = s["d_p2"]
    wh_nu = s["wh_nu"]; wh_Lr = s["wh_Lr"]; wh_q1 = s["wh_q1"]
    wh_q2 = s["wh_q2"]; wh_p1 = s["wh_p1"]; wh_p2 = s["wh_p2"]
    GAMMA_nu = s["GAMMA_nu"]; GAMMA_Lr = s["GAMMA_Lr"]
    GAMMA_q1 = s["GAMMA_q1"]; GAMMA_q2 = s["GAMMA_q2"]
    GAMMA_p1 = s["GAMMA_p1"]; GAMMA_p2 = s["GAMMA_p2"]

    # xi partials
    xi1_nu = s["xi1_nu"]; xi1_Lr = s["xi1_Lr"]; xi1_q1 = s["xi1_q1"]
    xi1_q2 = s["xi1_q2"]; xi1_p1 = s["xi1_p1"]; xi1_p2 = s["xi1_p2"]
    xi2_nu = s["xi2_nu"]; xi2_Lr = s["xi2_Lr"]; xi2_q1 = s["xi2_q1"]
    xi2_q2 = s["xi2_q2"]; xi2_p1 = s["xi2_p1"]; xi2_p2 = s["xi2_p2"]

    # EOM partials
    q1p_nu = s["q1p_nu"]; q1p_Lr = s["q1p_Lr"]; q1p_q1 = s["q1p_q1"]
    q1p_q2 = s["q1p_q2"]; q1p_p1 = s["q1p_p1"]; q1p_p2 = s["q1p_p2"]
    q2p_nu = s["q2p_nu"]; q2p_Lr = s["q2p_Lr"]; q2p_q1 = s["q2p_q1"]
    q2p_q2 = s["q2p_q2"]; q2p_p1 = s["q2p_p1"]; q2p_p2 = s["q2p_p2"]
    p1p_nu = s["p1p_nu"]; p1p_Lr = s["p1p_Lr"]; p1p_q1 = s["p1p_q1"]
    p1p_q2 = s["p1p_q2"]; p1p_p1 = s["p1p_p1"]; p1p_p2 = s["p1p_p2"]
    p2p_nu = s["p2p_nu"]; p2p_Lr = s["p2p_Lr"]; p2p_q1 = s["p2p_q1"]
    p2p_q2 = s["p2p_q2"]; p2p_p1 = s["p2p_p1"]; p2p_p2 = s["p2p_p2"]
    Lrp_nu = s["Lrp_nu"]; Lrp_Lr = s["Lrp_Lr"]; Lrp_q1 = s["Lrp_q1"]
    Lrp_q2 = s["Lrp_q2"]; Lrp_p1 = s["Lrp_p1"]; Lrp_p2 = s["Lrp_p2"]

    # rpn partials
    rpn_nu = s["rpn_nu"]; rpn_Lr = s["rpn_Lr"]; rpn_q1 = s["rpn_q1"]
    rpn_q2 = s["rpn_q2"]; rpn_p1 = s["rpn_p1"]; rpn_p2 = s["rpn_p2"]

    # Inverse-derivative param vectors
    r_nu_vector = s["r_nu_vector"]; r_Lr_vector = s["r_Lr_vector"]
    r_q1_vector = s["r_q1_vector"]; r_q2_vector = s["r_q2_vector"]
    r_p1_vector = s["r_p1_vector"]; r_p2_vector = s["r_p2_vector"]
    r2_nu_vector = s["r2_nu_vector"]; r2_Lr_vector = s["r2_Lr_vector"]
    r2_q1_vector = s["r2_q1_vector"]; r2_q2_vector = s["r2_q2_vector"]
    r2_p1_vector = s["r2_p1_vector"]; r2_p2_vector = s["r2_p2_vector"]
    r3_nu_vector = s["r3_nu_vector"]; r3_Lr_vector = s["r3_Lr_vector"]
    r3_q1_vector = s["r3_q1_vector"]; r3_q2_vector = s["r3_q2_vector"]
    r3_p1_vector = s["r3_p1_vector"]; r3_p2_vector = s["r3_p2_vector"]
    D_nu = s["D_nu"]; D_Lr = s["D_Lr"]; D_q1 = s["D_q1"]
    D_q2 = s["D_q2"]; D_p1 = s["D_p1"]; D_p2 = s["D_p2"]
    fiD_nu = s["fiD_nu"]; fiD_Lr = s["fiD_Lr"]; fiD_q1 = s["fiD_q1"]
    fiD_q2 = s["fiD_q2"]; fiD_p1 = s["fiD_p1"]; fiD_p2 = s["fiD_p2"]
    C_nu = s["C_nu"]; C_Lr = s["C_Lr"]; C_q1 = s["C_q1"]
    C_q2 = s["C_q2"]; C_p1 = s["C_p1"]; C_p2 = s["C_p2"]

    # Higher-order inverse derivative values
    fir3_p_nu = s["fir3_p_nu"]; fir3_p_Lr = s["fir3_p_Lr"]
    fir3_p_q1 = s["fir3_p_q1"]; fir3_p_q2 = s["fir3_p_q2"]
    fir3_p_p1 = s["fir3_p_p1"]; fir3_p_p2 = s["fir3_p_p2"]

    fir_nu = s["fir_nu"]; fir_Lr = s["fir_Lr"]; fir_q1 = s["fir_q1"]
    fir_q2 = s["fir_q2"]; fir_p1 = s["fir_p1"]; fir_p2 = s["fir_p2"]
    firp_nu = s["firp_nu"]; firp_Lr = s["firp_Lr"]; firp_q1 = s["firp_q1"]
    firp_q2 = s["firp_q2"]; firp_p1 = s["firp_p1"]; firp_p2 = s["firp_p2"]
    fir2_nu = s["fir2_nu"]; fir2_Lr = s["fir2_Lr"]; fir2_q1 = s["fir2_q1"]
    fir2_q2 = s["fir2_q2"]; fir2_p1 = s["fir2_p1"]; fir2_p2 = s["fir2_p2"]
    fir2p_nu = s["fir2p_nu"]; fir2p_Lr = s["fir2p_Lr"]; fir2p_q1 = s["fir2p_q1"]
    fir2p_q2 = s["fir2p_q2"]; fir2p_p1 = s["fir2p_p1"]; fir2p_p2 = s["fir2p_p2"]
    fir3_nu = s["fir3_nu"]; fir3_Lr = s["fir3_Lr"]; fir3_q1 = s["fir3_q1"]
    fir3_q2 = s["fir3_q2"]; fir3_p1 = s["fir3_p1"]; fir3_p2 = s["fir3_p2"]

    f2rp_nu = s["f2rp_nu"]; f2rp_Lr = s["f2rp_Lr"]; f2rp_q1 = s["f2rp_q1"]
    f2rp_q2 = s["f2rp_q2"]; f2rp_p1 = s["f2rp_p1"]; f2rp_p2 = s["f2rp_p2"]

    # NOTE: STM partial accumulators are NOT read here -- they are
    # dt-dependent and belong in evaluate_order_2.

    # ================================================================== #
    # SECOND ORDER DERIVATIVES (legacy lines 674-1218)                    #
    # ================================================================== #

    # # INTERMEDIATE COMPUTATIONS

    # qs  = (1+q1s+q2s)
    # ps  = (1+p1s+p2s)

    qs_nu = 0; qs_Lr = 0; qs_p1 = 0; qs_p2 = 0
    qs_q1 = 2*q1_0; qs_q2 = 2*q2_0

    qsp = 2*(q1_0*q1p_0+q2_0*q2p_0)

    qsp_nu = 2*(q1_0*q1p_nu+q2_0*q2p_nu)
    qsp_Lr = 2*(q1_0*q1p_Lr+q2_0*q2p_Lr)
    qsp_p1 = 2*(q1_0*q1p_p1+q2_0*q2p_p1)
    qsp_p2 = 2*(q1_0*q1p_p2+q2_0*q2p_p2)
    qsp_q1 = 2*(q1p_0+q1_0*q1p_q1+q2_0*q2p_q1)
    qsp_q2 = 2*(q1_0*q1p_q2+q2p_0+q2_0*q2p_q2)

    psp = 2*(p1_0*p1p_0+p2_0*p2p_0)

    psp_nu = 2*(p1_0*p1p_nu+p2_0*p2p_nu)
    psp_Lr = 2*(p1_0*p1p_Lr+p2_0*p2p_Lr)
    psp_p1 = 2*(p1p_0+p1_0*p1p_p1+p2_0*p2p_p1)
    psp_p2 = 2*(p1_0*p1p_p2+p2p_0+p2_0*p2p_p2)
    psp_q1 = 2*(p1_0*p1p_q1+p2_0*p2p_q1)
    psp_q2 = 2*(p1_0*p1p_q2+p2_0*p2p_q2)

    # hr
    hr_nu = h_nu*fir + h*fir_nu
    hr_Lr = h_Lr*fir + h*fir_Lr
    hr_q1 = h_q1*fir + h*fir_q1
    hr_q2 = h_q2*fir + h*fir_q2
    hr_p1 = h_p1*fir + h*fir_p1
    hr_p2 = h_p2*fir + h*fir_p2

    # Xp
    Xp_nu = rp_nu*cosL + rp*cosL_nu - hr_nu*sinL - hr*sinL_nu
    Xp_Lr = rp_Lr*cosL + rp*cosL_Lr - hr_Lr*sinL - hr*sinL_Lr
    Xp_q1 = rp_q1*cosL + rp*cosL_q1 - hr_q1*sinL - hr*sinL_q1
    Xp_q2 = rp_q2*cosL + rp*cosL_q2 - hr_q2*sinL - hr*sinL_q2
    Xp_p1 = rp_p1*cosL + rp*cosL_p1 - hr_p1*sinL - hr*sinL_p1
    Xp_p2 = rp_p2*cosL + rp*cosL_p2 - hr_p2*sinL - hr*sinL_p2

    # Yp
    Yp_nu = rp_nu*sinL + rp*sinL_nu + hr_nu*cosL + hr*cosL_nu
    Yp_Lr = rp_Lr*sinL + rp*sinL_Lr + hr_Lr*cosL + hr*cosL_Lr
    Yp_q1 = rp_q1*sinL + rp*sinL_q1 + hr_q1*cosL + hr*cosL_q1
    Yp_q2 = rp_q2*sinL + rp*sinL_q2 + hr_q2*cosL + hr*cosL_q2
    Yp_p1 = rp_p1*sinL + rp*sinL_p1 + hr_p1*cosL + hr*cosL_p1
    Yp_p2 = rp_p2*sinL + rp*sinL_p2 + hr_p2*cosL + hr*cosL_p2

    # zgp
    Cp  = 2*(Yp*q2_0 + Y*q2p_0 - Xp*q1_0 - X*q1p_0)
    Dp  = rp*qs + r*qsp
    Delta_vector = np.array([D, Dp])

    fiDp  = derivatives_of_inverse(Delta_vector,True)
    zgp = Cp*fiD + C*fiDp

    Cp_nu = 2*(Yp_nu*q2_0 + Y_nu*q2p_0 + Y*q2p_nu - Xp_nu*q1_0 - X_nu*q1p_0 - X*q1p_nu)
    Cp_Lr = 2*(Yp_Lr*q2_0 + Y_Lr*q2p_0 + Y*q2p_Lr - Xp_Lr*q1_0 - X_Lr*q1p_0 - X*q1p_Lr)
    Cp_p1 = 2*(Yp_p1*q2_0 + Y_p1*q2p_0 + Y*q2p_p1 - Xp_p1*q1_0 - X_p1*q1p_0 - X*q1p_p1)
    Cp_p2 = 2*(Yp_p2*q2_0 + Y_p2*q2p_0 + Y*q2p_p2 - Xp_p2*q1_0 - X_p2*q1p_0 - X*q1p_p2)
    Cp_q1 = 2*(Yp_q1*q2_0 + Y_q1*q2p_0 + Y*q2p_q1 - Xp_q1*q1_0 - Xp - X_q1*q1p_0 - X*q1p_q1)
    Cp_q2 = 2*(Yp_q2*q2_0 + Yp + Y_q2*q2p_0 + Y*q2p_q2 - Xp_q2*q1_0 - X_q2*q1p_0 - X*q1p_q2)

    Dp_nu = rp_nu*qs + rp*qs_nu + r_nu*qsp + r*qsp_nu
    Dp_Lr = rp_Lr*qs + rp*qs_Lr + r_Lr*qsp + r*qsp_Lr
    Dp_p1 = rp_p1*qs + rp*qs_p1 + r_p1*qsp + r*qsp_p1
    Dp_p2 = rp_p2*qs + rp*qs_p2 + r_p2*qsp + r*qsp_p2
    Dp_q1 = rp_q1*qs + rp*qs_q1 + r_q1*qsp + r*qsp_q1
    Dp_q2 = rp_q2*qs + rp*qs_q2 + r_q2*qsp + r*qsp_q2

    D_nu_vector = np.array([D_nu, Dp_nu]); D_Lr_vector = np.array([D_Lr, Dp_Lr])
    D_q1_vector = np.array([D_q1, Dp_q1]); D_q2_vector = np.array([D_q2, Dp_q2])
    D_p1_vector = np.array([D_p1, Dp_p1]); D_p2_vector = np.array([D_p2, Dp_p2])

    fiDp_nu  = derivatives_of_inverse_wrt_param(Delta_vector,D_nu_vector,True)
    fiDp_Lr  = derivatives_of_inverse_wrt_param(Delta_vector,D_Lr_vector,True)
    fiDp_q1  = derivatives_of_inverse_wrt_param(Delta_vector,D_q1_vector,True)
    fiDp_q2  = derivatives_of_inverse_wrt_param(Delta_vector,D_q2_vector,True)
    fiDp_p1  = derivatives_of_inverse_wrt_param(Delta_vector,D_p1_vector,True)
    fiDp_p2  = derivatives_of_inverse_wrt_param(Delta_vector,D_p2_vector,True)

    zgp_nu = Cp_nu*fiD + Cp*fiD_nu + C_nu*fiDp + C*fiDp_nu
    zgp_Lr = Cp_Lr*fiD + Cp*fiD_Lr + C_Lr*fiDp + C*fiDp_Lr
    zgp_q1 = Cp_q1*fiD + Cp*fiD_q1 + C_q1*fiDp + C*fiDp_q1
    zgp_q2 = Cp_q2*fiD + Cp*fiD_q2 + C_q2*fiDp + C*fiDp_q2
    zgp_p1 = Cp_p1*fiD + Cp*fiD_p1 + C_p1*fiDp + C*fiDp_p1
    zgp_p2 = Cp_p2*fiD + Cp*fiD_p2 + C_p2*fiDp + C*fiDp_p2

    # U functions
    # fUz = (1-3*zg**2)
    fUzp = -6*zg*zgp

    fUzp_nu = -6*(zg_nu*zgp+zg*zgp_nu)
    fUzp_Lr = -6*(zg_Lr*zgp+zg*zgp_Lr)
    fUzp_q1 = -6*(zg_q1*zgp+zg*zgp_q1)
    fUzp_q2 = -6*(zg_q2*zgp+zg*zgp_q2)
    fUzp_p1 = -6*(zg_p1*zgp+zg*zgp_p1)
    fUzp_p2 = -6*(zg_p2*zgp+zg*zgp_p2)

    Up      = -A*( fUzp*fir3 + fUz*fir3p )

    Up_nu = -A*(fUzp_nu*fir3 + fUzp*fir3_nu + fUz_nu*fir3p + fUz*fir3_p_nu)
    Up_Lr = -A*(fUzp_Lr*fir3 + fUzp*fir3_Lr + fUz_Lr*fir3p + fUz*fir3_p_Lr)
    Up_q1 = -A*(fUzp_q1*fir3 + fUzp*fir3_q1 + fUz_q1*fir3p + fUz*fir3_p_q1)
    Up_q2 = -A*(fUzp_q2*fir3 + fUzp*fir3_q2 + fUz_q2*fir3p + fUz*fir3_p_q2)
    Up_p1 = -A*(fUzp_p1*fir3 + fUzp*fir3_p1 + fUz_p1*fir3p + fUz*fir3_p_p1)
    Up_p2 = -A*(fUzp_p2*fir3 + fUzp*fir3_p2 + fUz_p2*fir3p + fUz*fir3_p_p2)

    # beta functions
    bp  = -psp*fib/2

    beta_p1_vector = np.array([beta_p1]); beta_p2_vector = np.array([beta_p2])

    fib_nu = 0; fib_Lr = 0; fib_q1 = 0; fib_q2 = 0
    fib_p1  = derivatives_of_inverse_wrt_param(beta_vector,beta_p1_vector,True)
    fib_p2  = derivatives_of_inverse_wrt_param(beta_vector,beta_p2_vector,True)

    bp_nu = -psp_nu*fib/2
    bp_Lr = -psp_Lr*fib/2
    bp_q1 = -psp_q1*fib/2
    bp_q2 = -psp_q2*fib/2
    bp_p1 = -(psp_p1*fib + psp*fib_p1)/2
    bp_p2 = -(psp_p2*fib + psp*fib_p2)/2

    beta_vector = np.append(beta_vector, bp)
    fibp  = derivatives_of_inverse(beta_vector,True)

    beta_nu_vector = np.array([beta_nu, bp_nu]); beta_Lr_vector = np.array([beta_Lr, bp_Lr])
    beta_q1_vector = np.array([beta_q1, bp_q1]); beta_q2_vector = np.array([beta_q2, bp_q2])
    beta_p1_vector = np.array([beta_p1, bp_p1]); beta_p2_vector = np.array([beta_p2, bp_p2])

    fibp_nu  = derivatives_of_inverse_wrt_param(beta_vector,beta_nu_vector,True)
    fibp_Lr  = derivatives_of_inverse_wrt_param(beta_vector,beta_Lr_vector,True)
    fibp_q1  = derivatives_of_inverse_wrt_param(beta_vector,beta_q1_vector,True)
    fibp_q2  = derivatives_of_inverse_wrt_param(beta_vector,beta_q2_vector,True)
    fibp_p1  = derivatives_of_inverse_wrt_param(beta_vector,beta_p1_vector,True)
    fibp_p2  = derivatives_of_inverse_wrt_param(beta_vector,beta_p2_vector,True)

    # L functions
    cosLp   = Xp*fir + X*firp
    sinLp   = Yp*fir + Y*firp

    cosLp_nu    = Xp_nu*fir + Xp*fir_nu + X_nu*firp + X*firp_nu
    cosLp_Lr    = Xp_Lr*fir + Xp*fir_Lr + X_Lr*firp + X*firp_Lr
    cosLp_q1    = Xp_q1*fir + Xp*fir_q1 + X_q1*firp + X*firp_q1
    cosLp_q2    = Xp_q2*fir + Xp*fir_q2 + X_q2*firp + X*firp_q2
    cosLp_p1    = Xp_p1*fir + Xp*fir_p1 + X_p1*firp + X*firp_p1
    cosLp_p2    = Xp_p2*fir + Xp*fir_p2 + X_p2*firp + X*firp_p2

    sinLp_nu    = Yp_nu*fir + Yp*fir_nu + Y_nu*firp + Y*firp_nu
    sinLp_Lr    = Yp_Lr*fir + Yp*fir_Lr + Y_Lr*firp + Y*firp_Lr
    sinLp_q1    = Yp_q1*fir + Yp*fir_q1 + Y_q1*firp + Y*firp_q1
    sinLp_q2    = Yp_q2*fir + Yp*fir_q2 + Y_q2*firp + Y*firp_q2
    sinLp_p1    = Yp_p1*fir + Yp*fir_p1 + Y_p1*firp + Y*firp_p1
    sinLp_p2    = Yp_p2*fir + Yp*fir_p2 + Y_p2*firp + Y*firp_p2

    # c functions
    cp  = (mu_norm**2/nu_0)**(1/3)*bp
    c_vector    = np.append(c_vector, cp)

    fic_vector  = derivatives_of_inverse(c_vector)
    fic = fic_vector[0]; ficp = fic_vector[1]

    f2cp_vector = derivatives_of_product(c_vector)
    f2cp = f2cp_vector[0]

    cp_nu = mu_norm**(2/3) * ( -1/3/nu_0**(4/3)*bp + (1/nu_0)**(1/3)*bp_nu )
    cp_Lr = mu_norm**(2/3)/nu_0**(1/3)*bp_Lr
    cp_q1 = mu_norm**(2/3)/nu_0**(1/3)*bp_q1
    cp_q2 = mu_norm**(2/3)/nu_0**(1/3)*bp_q2
    cp_p1 = mu_norm**(2/3)/nu_0**(1/3)*bp_p1
    cp_p2 = mu_norm**(2/3)/nu_0**(1/3)*bp_p2

    c_nu_vector = np.array([c_nu, cp_nu]); c_Lr_vector = np.array([c_Lr, cp_Lr])
    c_q1_vector = np.array([c_q1, cp_q1]); c_q2_vector = np.array([c_q2, cp_q2])
    c_p1_vector = np.array([c_p1, cp_p1]); c_p2_vector = np.array([c_p2, cp_p2])

    ficp_nu  = derivatives_of_inverse_wrt_param(c_vector,c_nu_vector,True)
    ficp_Lr  = derivatives_of_inverse_wrt_param(c_vector,c_Lr_vector,True)
    ficp_q1  = derivatives_of_inverse_wrt_param(c_vector,c_q1_vector,True)
    ficp_q2  = derivatives_of_inverse_wrt_param(c_vector,c_q2_vector,True)
    ficp_p1  = derivatives_of_inverse_wrt_param(c_vector,c_p1_vector,True)
    ficp_p2  = derivatives_of_inverse_wrt_param(c_vector,c_p2_vector,True)

    f2cp_nu = derivatives_of_product_wrt_param(c_vector,c_nu_vector,True)
    f2cp_Lr = derivatives_of_product_wrt_param(c_vector,c_Lr_vector,True)
    f2cp_q1 = derivatives_of_product_wrt_param(c_vector,c_q1_vector,True)
    f2cp_q2 = derivatives_of_product_wrt_param(c_vector,c_q2_vector,True)
    f2cp_p1 = derivatives_of_product_wrt_param(c_vector,c_p1_vector,True)
    f2cp_p2 = derivatives_of_product_wrt_param(c_vector,c_p2_vector,True)

    # r functions
    rpn_p   = p2p_0*sinL + p2_0*sinLp - ( p1p_0*cosL + p1_0*cosLp )

    rpn_p_nu = p2p_nu*sinL + p2p_0*sinL_nu + p2_0*sinLp_nu - (p1p_nu*cosL+p1p_0*cosL_nu+p1_0*cosLp_nu)
    rpn_p_Lr = p2p_Lr*sinL + p2p_0*sinL_Lr + p2_0*sinLp_Lr - (p1p_Lr*cosL+p1p_0*cosL_Lr+p1_0*cosLp_Lr)
    rpn_p_q1 = p2p_q1*sinL + p2p_0*sinL_q1 + p2_0*sinLp_q1 - (p1p_q1*cosL+p1p_0*cosL_q1+p1_0*cosLp_q1)
    rpn_p_q2 = p2p_q2*sinL + p2p_0*sinL_q2 + p2_0*sinLp_q2 - (p1p_q2*cosL+p1p_0*cosL_q2+p1_0*cosLp_q2)
    rpn_p_p1 = p2p_p1*sinL + p2p_0*sinL_p1 + p2_0*sinLp_p1 - (p1p_p1*cosL+p1p_0*cosL_p1+cosLp+p1_0*cosLp_p1)
    rpn_p_p2 = p2p_p2*sinL + p2p_0*sinL_p2 + sinLp + p2_0*sinLp_p2 - (p1p_p2*cosL+p1p_0*cosL_p2+p1_0*cosLp_p2)

    rpp     = mu_norm * ( rpn_p*fic + rpn*ficp)
    r_vector = np.append(r_vector, rpp)
    firp2   = derivatives_of_inverse(r_vector,True)
    f2rpp  = derivatives_of_product(r_vector,True)

    r2_vector   = np.append(r2_vector, 2*f2rpp)
    fir2p2      = derivatives_of_inverse(r2_vector,True)

    r3p2        = 3*(2*r*rp**2 + r2*rpp)
    r3_vector   = np.append(r3_vector, r3p2)
    fir3p2 = derivatives_of_inverse(r3_vector,True)

    rpp_nu = mu_norm * (rpn_p_nu*fic + rpn_p*fic_nu + rpn_nu*ficp + rpn*ficp_nu)
    rpp_Lr = mu_norm * (rpn_p_Lr*fic + rpn_p*fic_Lr + rpn_Lr*ficp + rpn*ficp_Lr)
    rpp_q1 = mu_norm * (rpn_p_q1*fic + rpn_p*fic_q1 + rpn_q1*ficp + rpn*ficp_q1)
    rpp_q2 = mu_norm * (rpn_p_q2*fic + rpn_p*fic_q2 + rpn_q2*ficp + rpn*ficp_q2)
    rpp_p1 = mu_norm * (rpn_p_p1*fic + rpn_p*fic_p1 + rpn_p1*ficp + rpn*ficp_p1)
    rpp_p2 = mu_norm * (rpn_p_p2*fic + rpn_p*fic_p2 + rpn_p2*ficp + rpn*ficp_p2)

    r_nu_vector = np.append(r_nu_vector, rpp_nu); r_Lr_vector = np.append(r_Lr_vector, rpp_Lr)
    r_q1_vector = np.append(r_q1_vector, rpp_q1); r_q2_vector = np.append(r_q2_vector, rpp_q2)
    r_p1_vector = np.append(r_p1_vector, rpp_p1); r_p2_vector = np.append(r_p2_vector, rpp_p2)

    firp2_nu   = derivatives_of_inverse_wrt_param(r_vector,r_nu_vector,True)
    firp2_Lr   = derivatives_of_inverse_wrt_param(r_vector,r_Lr_vector,True)
    firp2_q1   = derivatives_of_inverse_wrt_param(r_vector,r_q1_vector,True)
    firp2_q2   = derivatives_of_inverse_wrt_param(r_vector,r_q2_vector,True)
    firp2_p1   = derivatives_of_inverse_wrt_param(r_vector,r_p1_vector,True)
    firp2_p2   = derivatives_of_inverse_wrt_param(r_vector,r_p2_vector,True)

    f2rpp_nu  = derivatives_of_product_wrt_param(r_vector,r_nu_vector,True)
    f2rpp_Lr  = derivatives_of_product_wrt_param(r_vector,r_Lr_vector,True)
    f2rpp_q1  = derivatives_of_product_wrt_param(r_vector,r_q1_vector,True)
    f2rpp_q2  = derivatives_of_product_wrt_param(r_vector,r_q2_vector,True)
    f2rpp_p1  = derivatives_of_product_wrt_param(r_vector,r_p1_vector,True)
    f2rpp_p2  = derivatives_of_product_wrt_param(r_vector,r_p2_vector,True)

    r3p2_nu = 3*(2*r_nu*rp**2 + 4*r*rp*rp_nu + 2*r*r_nu*rpp + r2*rpp_nu)
    r3p2_Lr = 3*(2*r_Lr*rp**2 + 4*r*rp*rp_Lr + 2*r*r_Lr*rpp + r2*rpp_Lr)
    r3p2_q1 = 3*(2*r_q1*rp**2 + 4*r*rp*rp_q1 + 2*r*r_q1*rpp + r2*rpp_q1)
    r3p2_q2 = 3*(2*r_q2*rp**2 + 4*r*rp*rp_q2 + 2*r*r_q2*rpp + r2*rpp_q2)
    r3p2_p1 = 3*(2*r_p1*rp**2 + 4*r*rp*rp_p1 + 2*r*r_p1*rpp + r2*rpp_p1)
    r3p2_p2 = 3*(2*r_p2*rp**2 + 4*r*rp*rp_p2 + 2*r*r_p2*rpp + r2*rpp_p2)

    r3_nu_vector = np.append(r3_nu_vector, r3p2_nu); r3_Lr_vector = np.append(r3_Lr_vector, r3p2_Lr)
    r3_q1_vector = np.append(r3_q1_vector, r3p2_q1); r3_q2_vector = np.append(r3_q2_vector, r3p2_q2)
    r3_p1_vector = np.append(r3_p1_vector, r3p2_p1); r3_p2_vector = np.append(r3_p2_vector, r3p2_p2)

    fir3p2_nu  = derivatives_of_inverse_wrt_param(r3_vector,r3_nu_vector,True)
    fir3p2_Lr  = derivatives_of_inverse_wrt_param(r3_vector,r3_Lr_vector,True)
    fir3p2_q1  = derivatives_of_inverse_wrt_param(r3_vector,r3_q1_vector,True)
    fir3p2_q2  = derivatives_of_inverse_wrt_param(r3_vector,r3_q2_vector,True)
    fir3p2_p1  = derivatives_of_inverse_wrt_param(r3_vector,r3_p1_vector,True)
    fir3p2_p2  = derivatives_of_inverse_wrt_param(r3_vector,r3_p2_vector,True)

    r2_nu_vector = np.append(r2_nu_vector, 2*f2rpp_nu); r2_Lr_vector = np.append(r2_Lr_vector, 2*f2rpp_Lr)
    r2_q1_vector = np.append(r2_q1_vector, 2*f2rpp_q1); r2_q2_vector = np.append(r2_q2_vector, 2*f2rpp_q2)
    r2_p1_vector = np.append(r2_p1_vector, 2*f2rpp_p1); r2_p2_vector = np.append(r2_p2_vector, 2*f2rpp_p2)

    fir2p2_nu = derivatives_of_inverse_wrt_param(r2_vector,r2_nu_vector,True)
    fir2p2_Lr = derivatives_of_inverse_wrt_param(r2_vector,r2_Lr_vector,True)
    fir2p2_q1 = derivatives_of_inverse_wrt_param(r2_vector,r2_q1_vector,True)
    fir2p2_q2 = derivatives_of_inverse_wrt_param(r2_vector,r2_q2_vector,True)
    fir2p2_p1 = derivatives_of_inverse_wrt_param(r2_vector,r2_p1_vector,True)
    fir2p2_p2 = derivatives_of_inverse_wrt_param(r2_vector,r2_p2_vector,True)

    # h function
    hp      = (f2cp - 2*f2rp*U - r2*Up)*fih
    h_vector = np.append(h_vector, hp)
    fihp    = derivatives_of_inverse(h_vector,True)

    hp_nu = (f2cp_nu - 2*f2rp_nu*U - 2*f2rp*U_nu - 2*r*r_nu*Up - r2*Up_nu)*fih + \
            (f2cp - 2*f2rp*U - r2*Up)*fih_nu
    hp_Lr = (f2cp_Lr - 2*f2rp_Lr*U - 2*f2rp*U_Lr - 2*r*r_Lr*Up - r2*Up_Lr)*fih + \
            (f2cp - 2*f2rp*U - r2*Up)*fih_Lr
    hp_q1 = (f2cp_q1 - 2*f2rp_q1*U - 2*f2rp*U_q1 - 2*r*r_q1*Up - r2*Up_q1)*fih + \
            (f2cp - 2*f2rp*U - r2*Up)*fih_q1
    hp_q2 = (f2cp_q2 - 2*f2rp_q2*U - 2*f2rp*U_q2 - 2*r*r_q2*Up - r2*Up_q2)*fih + \
            (f2cp - 2*f2rp*U - r2*Up)*fih_q2
    hp_p1 = (f2cp_p1 - 2*f2rp_p1*U - 2*f2rp*U_p1 - 2*r*r_p1*Up - r2*Up_p1)*fih + \
            (f2cp - 2*f2rp*U - r2*Up)*fih_p1
    hp_p2 = (f2cp_p2 - 2*f2rp_p2*U - 2*f2rp*U_p2 - 2*r*r_p2*Up - r2*Up_p2)*fih + \
            (f2cp - 2*f2rp*U - r2*Up)*fih_p2

    h_nu_vector = np.array([h_nu, hp_nu]); h_Lr_vector = np.array([h_Lr, hp_Lr])
    h_q1_vector = np.array([h_q1, hp_q1]); h_q2_vector = np.array([h_q2, hp_q2])
    h_p1_vector = np.array([h_p1, hp_p1]); h_p2_vector = np.array([h_p2, hp_p2])

    fihp_nu = derivatives_of_inverse_wrt_param(h_vector,h_nu_vector,True)
    fihp_Lr = derivatives_of_inverse_wrt_param(h_vector,h_Lr_vector,True)
    fihp_q1 = derivatives_of_inverse_wrt_param(h_vector,h_q1_vector,True)
    fihp_q2 = derivatives_of_inverse_wrt_param(h_vector,h_q2_vector,True)
    fihp_p1 = derivatives_of_inverse_wrt_param(h_vector,h_p1_vector,True)
    fihp_p2 = derivatives_of_inverse_wrt_param(h_vector,h_p2_vector,True)

    # beta+1
    bm1_vector  = np.append(bm1_vector, bp)
    fibm1_p     = derivatives_of_inverse(bm1_vector,True)

    bm1_nu_vector = np.array([beta_nu, bp_nu])
    bm1_Lr_vector = np.array([beta_Lr, bp_Lr])
    bm1_q1_vector = np.array([beta_q1, bp_q1])
    bm1_q2_vector = np.array([beta_q2, bp_q2])
    bm1_p1_vector = np.array([beta_p1, bp_p1])
    bm1_p2_vector = np.array([beta_p2, bp_p2])

    fibm1_p_nu     = derivatives_of_inverse_wrt_param(bm1_vector,bm1_nu_vector,True)
    fibm1_p_Lr     = derivatives_of_inverse_wrt_param(bm1_vector,bm1_Lr_vector,True)
    fibm1_p_q1     = derivatives_of_inverse_wrt_param(bm1_vector,bm1_q1_vector,True)
    fibm1_p_q2     = derivatives_of_inverse_wrt_param(bm1_vector,bm1_q2_vector,True)
    fibm1_p_p1     = derivatives_of_inverse_wrt_param(bm1_vector,bm1_p1_vector,True)
    fibm1_p_p2     = derivatives_of_inverse_wrt_param(bm1_vector,bm1_p2_vector,True)

    # alpha
    alphap  = fibm1_p
    alpha_vector    = np.append(alpha_vector, alphap)
    fialphap        = derivatives_of_inverse(alpha_vector,True)

    alphap_nu = fibm1_p_nu; alphap_Lr = fibm1_p_Lr
    alphap_q1 = fibm1_p_q1; alphap_q2 = fibm1_p_q2
    alphap_p1 = fibm1_p_p1; alphap_p2 = fibm1_p_p2

    alpha_nu_vector = np.array([alpha_nu, alphap_nu]); alpha_Lr_vector = np.array([alpha_Lr, alphap_Lr])
    alpha_q1_vector = np.array([alpha_q1, alphap_q1]); alpha_q2_vector = np.array([alpha_q2, alphap_q2])
    alpha_p1_vector = np.array([alpha_p1, alphap_p1]); alpha_p2_vector = np.array([alpha_p2, alphap_p2])

    fialphap_nu    = derivatives_of_inverse_wrt_param(alpha_vector,alpha_nu_vector,True)
    fialphap_Lr    = derivatives_of_inverse_wrt_param(alpha_vector,alpha_Lr_vector,True)
    fialphap_q1    = derivatives_of_inverse_wrt_param(alpha_vector,alpha_q1_vector,True)
    fialphap_q2    = derivatives_of_inverse_wrt_param(alpha_vector,alpha_q2_vector,True)
    fialphap_p1    = derivatives_of_inverse_wrt_param(alpha_vector,alpha_p1_vector,True)
    fialphap_p2    = derivatives_of_inverse_wrt_param(alpha_vector,alpha_p2_vector,True)

    # delta
    deltap  = -qsp

    deltap_nu = -qsp_nu; deltap_Lr = -qsp_Lr
    deltap_q1 = -qsp_q1; deltap_q2 = -qsp_q2
    deltap_p1 = -qsp_p1; deltap_p2 = -qsp_p2

    # hr3
    hr3p    = hp*r3 + 3*h*r2*rp
    hr3_vector      = np.array([hr3, hr3p])
    fihr3, fihr3p    = derivatives_of_inverse(hr3_vector)

    hr3p_nu = hp_nu*r3 + hp*r3_nu + 3*h_nu*r2*rp + 3*h*(r2_nu*rp+r2*rp_nu)
    hr3p_Lr = hp_Lr*r3 + hp*r3_Lr + 3*h_Lr*r2*rp + 3*h*(r2_Lr*rp+r2*rp_Lr)
    hr3p_q1 = hp_q1*r3 + hp*r3_q1 + 3*h_q1*r2*rp + 3*h*(r2_q1*rp+r2*rp_q1)
    hr3p_q2 = hp_q2*r3 + hp*r3_q2 + 3*h_q2*r2*rp + 3*h*(r2_q2*rp+r2*rp_q2)
    hr3p_p1 = hp_p1*r3 + hp*r3_p1 + 3*h_p1*r2*rp + 3*h*(r2_p1*rp+r2*rp_p1)
    hr3p_p2 = hp_p2*r3 + hp*r3_p2 + 3*h_p2*r2*rp + 3*h*(r2_p2*rp+r2*rp_p2)

    hr3_nu_vector = np.array([hr3_nu, hr3p_nu]); hr3_Lr_vector = np.array([hr3_Lr, hr3p_Lr])
    hr3_q1_vector = np.array([hr3_q1, hr3p_q1]); hr3_q2_vector = np.array([hr3_q2, hr3p_q2])
    hr3_p1_vector = np.array([hr3_p1, hr3p_p1]); hr3_p2_vector = np.array([hr3_p2, hr3p_p2])

    fihr3p_nu    = derivatives_of_inverse_wrt_param(hr3_vector,hr3_nu_vector,True)
    fihr3p_Lr    = derivatives_of_inverse_wrt_param(hr3_vector,hr3_Lr_vector,True)
    fihr3p_q1    = derivatives_of_inverse_wrt_param(hr3_vector,hr3_q1_vector,True)
    fihr3p_q2    = derivatives_of_inverse_wrt_param(hr3_vector,hr3_q2_vector,True)
    fihr3p_p1    = derivatives_of_inverse_wrt_param(hr3_vector,hr3_p1_vector,True)
    fihr3p_p2    = derivatives_of_inverse_wrt_param(hr3_vector,hr3_p2_vector,True)

    # Gamma
    GAMMAp_ = fialphap + alphap*(1-r/a) - alpha*rp/a
    Ip      = 3*A*( (zgp*delta+zg*deltap)*fihr3 + zg*delta*fihr3p )
    dp      = (hp-cp)*fir2 + (h-c)*fir2p
    whp     = Ip*zg + I*zgp

    GAMMAp_nu = fialphap_nu + alphap_nu*(1-r/a) + alphap*(-r_nu/a + r/a**2*a_nu) - alpha_nu*rp/a - alpha*(rp_nu/a - rp/a**2*a_nu)
    GAMMAp_Lr = fialphap_Lr + alphap_Lr*(1-r/a) - alphap*r_Lr/a - alpha_Lr*rp/a - alpha*rp_Lr/a
    GAMMAp_q1 = fialphap_q1 + alphap_q1*(1-r/a) - alphap*r_q1/a - alpha_q1*rp/a - alpha*rp_q1/a
    GAMMAp_q2 = fialphap_q2 + alphap_q2*(1-r/a) - alphap*r_q2/a - alpha_q2*rp/a - alpha*rp_q2/a
    GAMMAp_p1 = fialphap_p1 + alphap_p1*(1-r/a) - alphap*r_p1/a - alpha_p1*rp/a - alpha*rp_p1/a
    GAMMAp_p2 = fialphap_p2 + alphap_p2*(1-r/a) - alphap*r_p2/a - alpha_p2*rp/a - alpha*rp_p2/a

    Ip_nu = 3*A*( (zgp_nu*delta+zgp*delta_nu+zg_nu*deltap+zg*deltap_nu)*fihr3 + (zgp*delta+zg*deltap)*fihr3_nu + \
                    (zg_nu*delta+zg*delta_nu)*fihr3p + zg*delta*fihr3p_nu )
    Ip_Lr = 3*A*( (zgp_Lr*delta+zgp*delta_Lr+zg_Lr*deltap+zg*deltap_Lr)*fihr3 + (zgp*delta+zg*deltap)*fihr3_Lr + \
                    (zg_Lr*delta+zg*delta_Lr)*fihr3p + zg*delta*fihr3p_Lr )
    Ip_q1 = 3*A*( (zgp_q1*delta+zgp*delta_q1+zg_q1*deltap+zg*deltap_q1)*fihr3 + (zgp*delta+zg*deltap)*fihr3_q1 + \
                    (zg_q1*delta+zg*delta_q1)*fihr3p + zg*delta*fihr3p_q1 )
    Ip_q2 = 3*A*( (zgp_q2*delta+zgp*delta_q2+zg_q2*deltap+zg*deltap_q2)*fihr3 + (zgp*delta+zg*deltap)*fihr3_q2 + \
                    (zg_q2*delta+zg*delta_q2)*fihr3p + zg*delta*fihr3p_q2 )
    Ip_p1 = 3*A*( (zgp_p1*delta+zgp*delta_p1+zg_p1*deltap+zg*deltap_p1)*fihr3 + (zgp*delta+zg*deltap)*fihr3_p1 + \
                    (zg_p1*delta+zg*delta_p1)*fihr3p + zg*delta*fihr3p_p1 )
    Ip_p2 = 3*A*( (zgp_p2*delta+zgp*delta_p2+zg_p2*deltap+zg*deltap_p2)*fihr3 + (zgp*delta+zg*deltap)*fihr3_p2 + \
                    (zg_p2*delta+zg*delta_p2)*fihr3p + zg*delta*fihr3p_p2 )

    dp_nu = (hp_nu-cp_nu)*fir2 + (hp-cp)*fir2_nu + (h_nu-c_nu)*fir2p + (h-c)*fir2p_nu
    dp_Lr = (hp_Lr-cp_Lr)*fir2 + (hp-cp)*fir2_Lr + (h_Lr-c_Lr)*fir2p + (h-c)*fir2p_Lr
    dp_q1 = (hp_q1-cp_q1)*fir2 + (hp-cp)*fir2_q1 + (h_q1-c_q1)*fir2p + (h-c)*fir2p_q1
    dp_q2 = (hp_q2-cp_q2)*fir2 + (hp-cp)*fir2_q2 + (h_q2-c_q2)*fir2p + (h-c)*fir2p_q2
    dp_p1 = (hp_p1-cp_p1)*fir2 + (hp-cp)*fir2_p1 + (h_p1-c_p1)*fir2p + (h-c)*fir2p_p1
    dp_p2 = (hp_p2-cp_p2)*fir2 + (hp-cp)*fir2_p2 + (h_p2-c_p2)*fir2p + (h-c)*fir2p_p2

    whp_nu = Ip_nu*zg + Ip*zg_nu + I_nu*zgp + I*zgp_nu
    whp_Lr = Ip_Lr*zg + Ip*zg_Lr + I_Lr*zgp + I*zgp_Lr
    whp_q1 = Ip_q1*zg + Ip*zg_q1 + I_q1*zgp + I*zgp_q1
    whp_q2 = Ip_q2*zg + Ip*zg_q2 + I_q2*zgp + I*zgp_q2
    whp_p1 = Ip_p1*zg + Ip*zg_p1 + I_p1*zgp + I*zgp_p1
    whp_p2 = Ip_p2*zg + Ip*zg_p2 + I_p2*zgp + I*zgp_p2

    xi1p    = Xp/a+2*p2p_0
    xi2p    = Yp/a+2*p1p_0

    xi1p_nu = Xp_nu/a - Xp/a**2*a_nu + 2*p2p_nu
    xi1p_Lr = Xp_Lr/a + 2*p2p_Lr
    xi1p_q1 = Xp_q1/a + 2*p2p_q1
    xi1p_q2 = Xp_q2/a + 2*p2p_q2
    xi1p_p1 = Xp_p1/a + 2*p2p_p1
    xi1p_p2 = Xp_p2/a + 2*p2p_p2

    xi2p_nu = Yp_nu/a - Yp/a**2*a_nu + 2*p1p_nu
    xi2p_Lr = Yp_Lr/a + 2*p1p_Lr
    xi2p_q1 = Yp_q1/a + 2*p1p_q1
    xi2p_q2 = Yp_q2/a + 2*p1p_q2
    xi2p_p1 = Yp_p1/a + 2*p1p_p1
    xi2p_p2 = Yp_p2/a + 2*p1p_p2

    # ELEMENT nu
    # nupp_0 = 0
    # ELEMENT p1
    p1pp_0  = p2p_0 * (d-wh) + p2_0 * (dp-whp) - U * ( ficp*xi1 + fic*xi1p ) - Up*fic*xi1
    # ELEMENT p2
    p2pp_0  = p1p_0 * (-d+wh) + p1_0 * (-dp+whp) + U * ( ficp*xi2 + fic*xi2p ) + Up*fic*xi2
    # ELEMENT Lr
    Lrpp_0  = dp - whp - U*(ficp*GAMMA_ + fic*GAMMAp_) - Up*fic*GAMMA_
    # ELEMENT q1
    q1pp_0  = - Ip*sinL - I*sinLp
    # ELEMENT q2
    q2pp_0  = - Ip*cosL - I*cosLp

    ctx.map_components[:, 1] = [0, q1pp_0, q2pp_0, p1pp_0, p2pp_0, Lrpp_0]

    # DERIVATIVES WRT INITIAL CONDITIONS (SECOND ORDER)
    # q1p2 derivatives
    q1p2_nu = -Ip_nu*sinL-Ip*sinL_nu -I_nu*sinLp-I*sinLp_nu
    q1p2_Lr = -Ip_Lr*sinL-Ip*sinL_Lr -I_Lr*sinLp-I*sinLp_Lr
    q1p2_q1 = -Ip_q1*sinL-Ip*sinL_q1 -I_q1*sinLp-I*sinLp_q1
    q1p2_q2 = -Ip_q2*sinL-Ip*sinL_q2 -I_q2*sinLp-I*sinLp_q2
    q1p2_p1 = -Ip_p1*sinL-Ip*sinL_p1 -I_p1*sinLp-I*sinLp_p1
    q1p2_p2 = -Ip_p2*sinL-Ip*sinL_p2 -I_p2*sinLp-I*sinLp_p2

    # q2p2 derivatives
    q2p2_nu = -Ip_nu*cosL-Ip*cosL_nu -I_nu*cosLp-I*cosLp_nu
    q2p2_Lr = -Ip_Lr*cosL-Ip*cosL_Lr -I_Lr*cosLp-I*cosLp_Lr
    q2p2_q1 = -Ip_q1*cosL-Ip*cosL_q1 -I_q1*cosLp-I*cosLp_q1
    q2p2_q2 = -Ip_q2*cosL-Ip*cosL_q2 -I_q2*cosLp-I*cosLp_q2
    q2p2_p1 = -Ip_p1*cosL-Ip*cosL_p1 -I_p1*cosLp-I*cosLp_p1
    q2p2_p2 = -Ip_p2*cosL-Ip*cosL_p2 -I_p2*cosLp-I*cosLp_p2

    # p1p2 derivatives
    p1p2_nu = p2p_nu*(d-wh)+p2p_0*(d_nu-wh_nu)+p2_0*(dp_nu-whp_nu)-U_nu*(ficp*xi1 + fic*xi1p)- \
                U*(ficp_nu*xi1+ficp*xi1_nu+fic_nu*xi1p+fic*xi1p_nu)-Up_nu*fic*xi1-Up*(fic_nu*xi1+fic*xi1_nu)
    p1p2_Lr = p2p_Lr*(d-wh)+p2p_0*(d_Lr-wh_Lr)+p2_0*(dp_Lr-whp_Lr)-U_Lr*(ficp*xi1 + fic*xi1p)- \
                U*(ficp_Lr*xi1+ficp*xi1_Lr+fic_Lr*xi1p+fic*xi1p_Lr)-Up_Lr*fic*xi1-Up*(fic_Lr*xi1+fic*xi1_Lr)
    p1p2_q1 = p2p_q1*(d-wh)+p2p_0*(d_q1-wh_q1)+p2_0*(dp_q1-whp_q1)-U_q1*(ficp*xi1 + fic*xi1p)- \
                U*(ficp_q1*xi1+ficp*xi1_q1+fic_q1*xi1p+fic*xi1p_q1)-Up_q1*fic*xi1-Up*(fic_q1*xi1+fic*xi1_q1)
    p1p2_q2 = p2p_q2*(d-wh)+p2p_0*(d_q2-wh_q2)+p2_0*(dp_q2-whp_q2)-U_q2*(ficp*xi1 + fic*xi1p)- \
                U*(ficp_q2*xi1+ficp*xi1_q2+fic_q2*xi1p+fic*xi1p_q2)-Up_q2*fic*xi1-Up*(fic_q2*xi1+fic*xi1_q2)
    p1p2_p1 = p2p_p1*(d-wh)+p2p_0*(d_p1-wh_p1)+p2_0*(dp_p1-whp_p1)-U_p1*(ficp*xi1 + fic*xi1p)- \
                U*(ficp_p1*xi1+ficp*xi1_p1+fic_p1*xi1p+fic*xi1p_p1)-Up_p1*fic*xi1-Up*(fic_p1*xi1+fic*xi1_p1)
    p1p2_p2 = p2p_p2*(d-wh)+p2p_0*(d_p2-wh_p2)+(dp-whp)+p2_0*(dp_p2-whp_p2)-U_p2*(ficp*xi1 + fic*xi1p)- \
                U*(ficp_p2*xi1+ficp*xi1_p2+fic_p2*xi1p+fic*xi1p_p2)-Up_p2*fic*xi1-Up*(fic_p2*xi1+fic*xi1_p2)

    # p2p2 derivatives
    p2p2_nu = p1p_nu*(-d+wh)+p1p_0*(-d_nu+wh_nu)+p1_0*(-dp_nu+whp_nu)+U_nu*(ficp*xi2 + fic*xi2p)+ \
                U*(ficp_nu*xi2+ficp*xi2_nu+fic_nu*xi2p+fic*xi2p_nu)+Up_nu*fic*xi2+Up*(fic_nu*xi2+fic*xi2_nu)
    p2p2_Lr = p1p_Lr*(-d+wh)+p1p_0*(-d_Lr+wh_Lr)+p1_0*(-dp_Lr+whp_Lr)+U_Lr*(ficp*xi2 + fic*xi2p)+ \
                U*(ficp_Lr*xi2+ficp*xi2_Lr+fic_Lr*xi2p+fic*xi2p_Lr)+Up_Lr*fic*xi2+Up*(fic_Lr*xi2+fic*xi2_Lr)
    p2p2_q1 = p1p_q1*(-d+wh)+p1p_0*(-d_q1+wh_q1)+p1_0*(-dp_q1+whp_q1)+U_q1*(ficp*xi2 + fic*xi2p)+ \
                U*(ficp_q1*xi2+ficp*xi2_q1+fic_q1*xi2p+fic*xi2p_q1)+Up_q1*fic*xi2+Up*(fic_q1*xi2+fic*xi2_q1)
    p2p2_q2 = p1p_q2*(-d+wh)+p1p_0*(-d_q2+wh_q2)+p1_0*(-dp_q2+whp_q2)+U_q2*(ficp*xi2 + fic*xi2p)+ \
                U*(ficp_q2*xi2+ficp*xi2_q2+fic_q2*xi2p+fic*xi2p_q2)+Up_q2*fic*xi2+Up*(fic_q2*xi2+fic*xi2_q2)
    p2p2_p1 = p1p_p1*(-d+wh)+p1p_0*(-d_p1+wh_p1)+(-dp+whp)+p1_0*(-dp_p1+whp_p1)+U_p1*(ficp*xi2 + fic*xi2p)+ \
                U*(ficp_p1*xi2+ficp*xi2_p1+fic_p1*xi2p+fic*xi2p_p1)+Up_p1*fic*xi2+Up*(fic_p1*xi2+fic*xi2_p1)
    p2p2_p2 = p1p_p2*(-d+wh)+p1p_0*(-d_p2+wh_p2)+p1_0*(-dp_p2+whp_p2)+U_p2*(ficp*xi2 + fic*xi2p)+ \
                U*(ficp_p2*xi2+ficp*xi2_p2+fic_p2*xi2p+fic*xi2p_p2)+Up_p2*fic*xi2+Up*(fic_p2*xi2+fic*xi2_p2)

    # Lrp2 derivatives
    Lrp2_nu = dp_nu - whp_nu - U_nu*(ficp*GAMMA_ + fic*GAMMAp_) - U*(ficp_nu*GAMMA_+ficp*GAMMA_nu+fic_nu*GAMMAp_+fic*GAMMAp_nu) - \
                Up_nu*fic*GAMMA_ - Up*(fic_nu*GAMMA_+fic*GAMMA_nu)
    Lrp2_Lr = dp_Lr - whp_Lr - U_Lr*(ficp*GAMMA_ + fic*GAMMAp_) - U*(ficp_Lr*GAMMA_+ficp*GAMMA_Lr+fic_Lr*GAMMAp_+fic*GAMMAp_Lr) - \
                Up_Lr*fic*GAMMA_ - Up*(fic_Lr*GAMMA_+fic*GAMMA_Lr)
    Lrp2_q1 = dp_q1 - whp_q1 - U_q1*(ficp*GAMMA_ + fic*GAMMAp_) - U*(ficp_q1*GAMMA_+ficp*GAMMA_q1+fic_q1*GAMMAp_+fic*GAMMAp_q1) - \
                Up_q1*fic*GAMMA_ - Up*(fic_q1*GAMMA_+fic*GAMMA_q1)
    Lrp2_q2 = dp_q2 - whp_q2 - U_q2*(ficp*GAMMA_ + fic*GAMMAp_) - U*(ficp_q2*GAMMA_+ficp*GAMMA_q2+fic_q2*GAMMAp_+fic*GAMMAp_q2) - \
                Up_q2*fic*GAMMA_ - Up*(fic_q2*GAMMA_+fic*GAMMA_q2)
    Lrp2_p1 = dp_p1 - whp_p1 - U_p1*(ficp*GAMMA_ + fic*GAMMAp_) - U*(ficp_p1*GAMMA_+ficp*GAMMA_p1+fic_p1*GAMMAp_+fic*GAMMAp_p1) - \
                Up_p1*fic*GAMMA_ - Up*(fic_p1*GAMMA_+fic*GAMMA_p1)
    Lrp2_p2 = dp_p2 - whp_p2 - U_p2*(ficp*GAMMA_ + fic*GAMMAp_) - U*(ficp_p2*GAMMA_+ficp*GAMMA_p2+fic_p2*GAMMAp_+fic*GAMMAp_p2) - \
                Up_p2*fic*GAMMA_ - Up*(fic_p2*GAMMA_+fic*GAMMA_p2)

    # ================================================================== #
    # Store everything back into scratch for order 3                      #
    # ================================================================== #

    # Updated vectors
    s["r_vector"] = r_vector; s["r2_vector"] = r2_vector; s["r3_vector"] = r3_vector
    s["beta_vector"] = beta_vector; s["c_vector"] = c_vector
    s["h_vector"] = h_vector; s["bm1_vector"] = bm1_vector
    s["alpha_vector"] = alpha_vector; s["Delta_vector"] = Delta_vector
    s["hr3_vector"] = hr3_vector

    # Updated param vectors
    s["r_nu_vector"] = r_nu_vector; s["r_Lr_vector"] = r_Lr_vector
    s["r_q1_vector"] = r_q1_vector; s["r_q2_vector"] = r_q2_vector
    s["r_p1_vector"] = r_p1_vector; s["r_p2_vector"] = r_p2_vector
    s["r2_nu_vector"] = r2_nu_vector; s["r2_Lr_vector"] = r2_Lr_vector
    s["r2_q1_vector"] = r2_q1_vector; s["r2_q2_vector"] = r2_q2_vector
    s["r2_p1_vector"] = r2_p1_vector; s["r2_p2_vector"] = r2_p2_vector
    s["r3_nu_vector"] = r3_nu_vector; s["r3_Lr_vector"] = r3_Lr_vector
    s["r3_q1_vector"] = r3_q1_vector; s["r3_q2_vector"] = r3_q2_vector
    s["r3_p1_vector"] = r3_p1_vector; s["r3_p2_vector"] = r3_p2_vector

    s["D_nu_vector"] = D_nu_vector; s["D_Lr_vector"] = D_Lr_vector
    s["D_q1_vector"] = D_q1_vector; s["D_q2_vector"] = D_q2_vector
    s["D_p1_vector"] = D_p1_vector; s["D_p2_vector"] = D_p2_vector

    s["beta_nu_vector"] = beta_nu_vector; s["beta_Lr_vector"] = beta_Lr_vector
    s["beta_q1_vector"] = beta_q1_vector; s["beta_q2_vector"] = beta_q2_vector
    s["beta_p1_vector"] = beta_p1_vector; s["beta_p2_vector"] = beta_p2_vector

    s["c_nu_vector"] = c_nu_vector; s["c_Lr_vector"] = c_Lr_vector
    s["c_q1_vector"] = c_q1_vector; s["c_q2_vector"] = c_q2_vector
    s["c_p1_vector"] = c_p1_vector; s["c_p2_vector"] = c_p2_vector

    s["h_nu_vector"] = h_nu_vector; s["h_Lr_vector"] = h_Lr_vector
    s["h_q1_vector"] = h_q1_vector; s["h_q2_vector"] = h_q2_vector
    s["h_p1_vector"] = h_p1_vector; s["h_p2_vector"] = h_p2_vector

    s["alpha_nu_vector"] = alpha_nu_vector; s["alpha_Lr_vector"] = alpha_Lr_vector
    s["alpha_q1_vector"] = alpha_q1_vector; s["alpha_q2_vector"] = alpha_q2_vector
    s["alpha_p1_vector"] = alpha_p1_vector; s["alpha_p2_vector"] = alpha_p2_vector

    s["hr3_nu_vector"] = hr3_nu_vector; s["hr3_Lr_vector"] = hr3_Lr_vector
    s["hr3_q1_vector"] = hr3_q1_vector; s["hr3_q2_vector"] = hr3_q2_vector
    s["hr3_p1_vector"] = hr3_p1_vector; s["hr3_p2_vector"] = hr3_p2_vector

    # New scalars computed by order 2
    s["Xp"] = Xp; s["Yp"] = Yp  # overwrite with partials-bearing versions
    s["cosLp"] = cosLp; s["sinLp"] = sinLp
    s["hp"] = hp; s["dp"] = dp; s["whp"] = whp
    s["Ip"] = Ip; s["Up"] = Up
    s["bp"] = bp; s["cp"] = cp
    s["alphap"] = alphap; s["deltap"] = deltap
    s["zgp"] = zgp; s["fUzp"] = fUzp
    s["GAMMAp_"] = GAMMAp_
    s["xi1p"] = xi1p; s["xi2p"] = xi2p
    s["qsp"] = qsp; s["psp"] = psp
    s["Cp"] = Cp; s["Dp"] = Dp
    s["rpn_p"] = rpn_p; s["rpp"] = rpp
    s["fiDp"] = fiDp
    s["fibp"] = fibp; s["ficp"] = ficp; s["fihp"] = fihp
    s["fialphap"] = fialphap
    s["fibm1_p"] = fibm1_p
    s["fihr3"] = fihr3; s["fihr3p"] = fihr3p
    s["firp2"] = firp2; s["fir2p2"] = fir2p2; s["fir3p2"] = fir3p2
    s["f2rpp"] = f2rpp; s["f2cp"] = f2cp
    s["r3p2"] = r3p2
    s["hr3p"] = hr3p

    # Partials of new order-2 intermediates
    s["Xp_nu"] = Xp_nu; s["Xp_Lr"] = Xp_Lr; s["Xp_q1"] = Xp_q1
    s["Xp_q2"] = Xp_q2; s["Xp_p1"] = Xp_p1; s["Xp_p2"] = Xp_p2
    s["Yp_nu"] = Yp_nu; s["Yp_Lr"] = Yp_Lr; s["Yp_q1"] = Yp_q1
    s["Yp_q2"] = Yp_q2; s["Yp_p1"] = Yp_p1; s["Yp_p2"] = Yp_p2

    s["cosLp_nu"] = cosLp_nu; s["cosLp_Lr"] = cosLp_Lr; s["cosLp_q1"] = cosLp_q1
    s["cosLp_q2"] = cosLp_q2; s["cosLp_p1"] = cosLp_p1; s["cosLp_p2"] = cosLp_p2
    s["sinLp_nu"] = sinLp_nu; s["sinLp_Lr"] = sinLp_Lr; s["sinLp_q1"] = sinLp_q1
    s["sinLp_q2"] = sinLp_q2; s["sinLp_p1"] = sinLp_p1; s["sinLp_p2"] = sinLp_p2

    s["zgp_nu"] = zgp_nu; s["zgp_Lr"] = zgp_Lr; s["zgp_q1"] = zgp_q1
    s["zgp_q2"] = zgp_q2; s["zgp_p1"] = zgp_p1; s["zgp_p2"] = zgp_p2

    s["fUzp_nu"] = fUzp_nu; s["fUzp_Lr"] = fUzp_Lr; s["fUzp_q1"] = fUzp_q1
    s["fUzp_q2"] = fUzp_q2; s["fUzp_p1"] = fUzp_p1; s["fUzp_p2"] = fUzp_p2

    s["Up_nu"] = Up_nu; s["Up_Lr"] = Up_Lr; s["Up_q1"] = Up_q1
    s["Up_q2"] = Up_q2; s["Up_p1"] = Up_p1; s["Up_p2"] = Up_p2

    s["bp_nu"] = bp_nu; s["bp_Lr"] = bp_Lr; s["bp_q1"] = bp_q1
    s["bp_q2"] = bp_q2; s["bp_p1"] = bp_p1; s["bp_p2"] = bp_p2

    s["cp_nu"] = cp_nu; s["cp_Lr"] = cp_Lr; s["cp_q1"] = cp_q1
    s["cp_q2"] = cp_q2; s["cp_p1"] = cp_p1; s["cp_p2"] = cp_p2

    s["hp_nu"] = hp_nu; s["hp_Lr"] = hp_Lr; s["hp_q1"] = hp_q1
    s["hp_q2"] = hp_q2; s["hp_p1"] = hp_p1; s["hp_p2"] = hp_p2

    s["alphap_nu"] = alphap_nu; s["alphap_Lr"] = alphap_Lr
    s["alphap_q1"] = alphap_q1; s["alphap_q2"] = alphap_q2
    s["alphap_p1"] = alphap_p1; s["alphap_p2"] = alphap_p2

    s["deltap_nu"] = deltap_nu; s["deltap_Lr"] = deltap_Lr
    s["deltap_q1"] = deltap_q1; s["deltap_q2"] = deltap_q2
    s["deltap_p1"] = deltap_p1; s["deltap_p2"] = deltap_p2

    s["Ip_nu"] = Ip_nu; s["Ip_Lr"] = Ip_Lr; s["Ip_q1"] = Ip_q1
    s["Ip_q2"] = Ip_q2; s["Ip_p1"] = Ip_p1; s["Ip_p2"] = Ip_p2

    s["dp_nu"] = dp_nu; s["dp_Lr"] = dp_Lr; s["dp_q1"] = dp_q1
    s["dp_q2"] = dp_q2; s["dp_p1"] = dp_p1; s["dp_p2"] = dp_p2

    s["whp_nu"] = whp_nu; s["whp_Lr"] = whp_Lr; s["whp_q1"] = whp_q1
    s["whp_q2"] = whp_q2; s["whp_p1"] = whp_p1; s["whp_p2"] = whp_p2

    s["GAMMAp_nu"] = GAMMAp_nu; s["GAMMAp_Lr"] = GAMMAp_Lr
    s["GAMMAp_q1"] = GAMMAp_q1; s["GAMMAp_q2"] = GAMMAp_q2
    s["GAMMAp_p1"] = GAMMAp_p1; s["GAMMAp_p2"] = GAMMAp_p2

    s["xi1p_nu"] = xi1p_nu; s["xi1p_Lr"] = xi1p_Lr; s["xi1p_q1"] = xi1p_q1
    s["xi1p_q2"] = xi1p_q2; s["xi1p_p1"] = xi1p_p1; s["xi1p_p2"] = xi1p_p2
    s["xi2p_nu"] = xi2p_nu; s["xi2p_Lr"] = xi2p_Lr; s["xi2p_q1"] = xi2p_q1
    s["xi2p_q2"] = xi2p_q2; s["xi2p_p1"] = xi2p_p1; s["xi2p_p2"] = xi2p_p2

    s["qsp_nu"] = qsp_nu; s["qsp_Lr"] = qsp_Lr; s["qsp_q1"] = qsp_q1
    s["qsp_q2"] = qsp_q2; s["qsp_p1"] = qsp_p1; s["qsp_p2"] = qsp_p2

    s["psp_nu"] = psp_nu; s["psp_Lr"] = psp_Lr; s["psp_q1"] = psp_q1
    s["psp_q2"] = psp_q2; s["psp_p1"] = psp_p1; s["psp_p2"] = psp_p2

    s["Cp_nu"] = Cp_nu; s["Cp_Lr"] = Cp_Lr; s["Cp_q1"] = Cp_q1
    s["Cp_q2"] = Cp_q2; s["Cp_p1"] = Cp_p1; s["Cp_p2"] = Cp_p2

    s["Dp_nu"] = Dp_nu; s["Dp_Lr"] = Dp_Lr; s["Dp_q1"] = Dp_q1
    s["Dp_q2"] = Dp_q2; s["Dp_p1"] = Dp_p1; s["Dp_p2"] = Dp_p2

    s["fiDp_nu"] = fiDp_nu; s["fiDp_Lr"] = fiDp_Lr; s["fiDp_q1"] = fiDp_q1
    s["fiDp_q2"] = fiDp_q2; s["fiDp_p1"] = fiDp_p1; s["fiDp_p2"] = fiDp_p2

    s["rpn_p_nu"] = rpn_p_nu; s["rpn_p_Lr"] = rpn_p_Lr; s["rpn_p_q1"] = rpn_p_q1
    s["rpn_p_q2"] = rpn_p_q2; s["rpn_p_p1"] = rpn_p_p1; s["rpn_p_p2"] = rpn_p_p2

    s["rpp_nu"] = rpp_nu; s["rpp_Lr"] = rpp_Lr; s["rpp_q1"] = rpp_q1
    s["rpp_q2"] = rpp_q2; s["rpp_p1"] = rpp_p1; s["rpp_p2"] = rpp_p2

    s["r3p2_nu"] = r3p2_nu; s["r3p2_Lr"] = r3p2_Lr; s["r3p2_q1"] = r3p2_q1
    s["r3p2_q2"] = r3p2_q2; s["r3p2_p1"] = r3p2_p1; s["r3p2_p2"] = r3p2_p2

    s["hr3p_nu"] = hr3p_nu; s["hr3p_Lr"] = hr3p_Lr; s["hr3p_q1"] = hr3p_q1
    s["hr3p_q2"] = hr3p_q2; s["hr3p_p1"] = hr3p_p1; s["hr3p_p2"] = hr3p_p2

    s["fihr3p_nu"] = fihr3p_nu; s["fihr3p_Lr"] = fihr3p_Lr
    s["fihr3p_q1"] = fihr3p_q1; s["fihr3p_q2"] = fihr3p_q2
    s["fihr3p_p1"] = fihr3p_p1; s["fihr3p_p2"] = fihr3p_p2

    s["firp2_nu"] = firp2_nu; s["firp2_Lr"] = firp2_Lr; s["firp2_q1"] = firp2_q1
    s["firp2_q2"] = firp2_q2; s["firp2_p1"] = firp2_p1; s["firp2_p2"] = firp2_p2

    s["fir2p2_nu"] = fir2p2_nu; s["fir2p2_Lr"] = fir2p2_Lr; s["fir2p2_q1"] = fir2p2_q1
    s["fir2p2_q2"] = fir2p2_q2; s["fir2p2_p1"] = fir2p2_p1; s["fir2p2_p2"] = fir2p2_p2

    s["fir3p2_nu"] = fir3p2_nu; s["fir3p2_Lr"] = fir3p2_Lr; s["fir3p2_q1"] = fir3p2_q1
    s["fir3p2_q2"] = fir3p2_q2; s["fir3p2_p1"] = fir3p2_p1; s["fir3p2_p2"] = fir3p2_p2

    s["f2rpp_nu"] = f2rpp_nu; s["f2rpp_Lr"] = f2rpp_Lr; s["f2rpp_q1"] = f2rpp_q1
    s["f2rpp_q2"] = f2rpp_q2; s["f2rpp_p1"] = f2rpp_p1; s["f2rpp_p2"] = f2rpp_p2

    s["f2cp_nu"] = f2cp_nu; s["f2cp_Lr"] = f2cp_Lr; s["f2cp_q1"] = f2cp_q1
    s["f2cp_q2"] = f2cp_q2; s["f2cp_p1"] = f2cp_p1; s["f2cp_p2"] = f2cp_p2

    s["ficp_nu"] = ficp_nu; s["ficp_Lr"] = ficp_Lr; s["ficp_q1"] = ficp_q1
    s["ficp_q2"] = ficp_q2; s["ficp_p1"] = ficp_p1; s["ficp_p2"] = ficp_p2

    s["fihp_nu"] = fihp_nu; s["fihp_Lr"] = fihp_Lr; s["fihp_q1"] = fihp_q1
    s["fihp_q2"] = fihp_q2; s["fihp_p1"] = fihp_p1; s["fihp_p2"] = fihp_p2

    s["fialphap_nu"] = fialphap_nu; s["fialphap_Lr"] = fialphap_Lr
    s["fialphap_q1"] = fialphap_q1; s["fialphap_q2"] = fialphap_q2
    s["fialphap_p1"] = fialphap_p1; s["fialphap_p2"] = fialphap_p2

    s["fib_nu"] = fib_nu; s["fib_Lr"] = fib_Lr; s["fib_q1"] = fib_q1
    s["fib_q2"] = fib_q2; s["fib_p1"] = fib_p1; s["fib_p2"] = fib_p2

    s["fibp_nu"] = fibp_nu; s["fibp_Lr"] = fibp_Lr; s["fibp_q1"] = fibp_q1
    s["fibp_q2"] = fibp_q2; s["fibp_p1"] = fibp_p1; s["fibp_p2"] = fibp_p2

    s["fibm1_p_nu"] = fibm1_p_nu; s["fibm1_p_Lr"] = fibm1_p_Lr
    s["fibm1_p_q1"] = fibm1_p_q1; s["fibm1_p_q2"] = fibm1_p_q2
    s["fibm1_p_p1"] = fibm1_p_p1; s["fibm1_p_p2"] = fibm1_p_p2

    s["hr_nu"] = hr_nu; s["hr_Lr"] = hr_Lr; s["hr_q1"] = hr_q1
    s["hr_q2"] = hr_q2; s["hr_p1"] = hr_p1; s["hr_p2"] = hr_p2

    s["qs_nu"] = qs_nu; s["qs_Lr"] = qs_Lr; s["qs_q1"] = qs_q1
    s["qs_q2"] = qs_q2; s["qs_p1"] = qs_p1; s["qs_p2"] = qs_p2

    # Updated fic (overwritten by c_vector inverse)
    s["fic"] = fic

    # Second-order EOM partials (stored for evaluate_order_2 and order 3)
    s["q1p2_nu"] = q1p2_nu; s["q1p2_Lr"] = q1p2_Lr; s["q1p2_q1"] = q1p2_q1
    s["q1p2_q2"] = q1p2_q2; s["q1p2_p1"] = q1p2_p1; s["q1p2_p2"] = q1p2_p2
    s["q2p2_nu"] = q2p2_nu; s["q2p2_Lr"] = q2p2_Lr; s["q2p2_q1"] = q2p2_q1
    s["q2p2_q2"] = q2p2_q2; s["q2p2_p1"] = q2p2_p1; s["q2p2_p2"] = q2p2_p2
    s["p1p2_nu"] = p1p2_nu; s["p1p2_Lr"] = p1p2_Lr; s["p1p2_q1"] = p1p2_q1
    s["p1p2_q2"] = p1p2_q2; s["p1p2_p1"] = p1p2_p1; s["p1p2_p2"] = p1p2_p2
    s["p2p2_nu"] = p2p2_nu; s["p2p2_Lr"] = p2p2_Lr; s["p2p2_q1"] = p2p2_q1
    s["p2p2_q2"] = p2p2_q2; s["p2p2_p1"] = p2p2_p1; s["p2p2_p2"] = p2p2_p2
    s["Lrp2_nu"] = Lrp2_nu; s["Lrp2_Lr"] = Lrp2_Lr; s["Lrp2_q1"] = Lrp2_q1
    s["Lrp2_q2"] = Lrp2_q2; s["Lrp2_p1"] = Lrp2_p1; s["Lrp2_p2"] = Lrp2_p2

    # Second-order EOM values (stored for evaluate_order_2 and order 3)
    s["p1pp_0"] = p1pp_0; s["p2pp_0"] = p2pp_0; s["Lrpp_0"] = Lrpp_0
    s["q1pp_0"] = q1pp_0; s["q2pp_0"] = q2pp_0


def evaluate_order_2(ctx: GEqOEPropagationContext) -> None:
    """Evaluate order-2 Taylor polynomial (dt-dependent).

    Calls :func:`evaluate_order_1` first to populate the order-1
    STM accumulators, then adds the order-2 polynomial terms to
    ``ctx.y_prop`` and the STM partial accumulators in ``ctx.scratch``.
    """
    from astrodyn_core.propagation.geqoe.taylor_order_1 import evaluate_order_1
    evaluate_order_1(ctx)

    s = ctx.scratch
    dt_norm = ctx.dt_norm

    # --- Read EOM coefficients from scratch ---
    q1pp_0 = s["q1pp_0"]; q2pp_0 = s["q2pp_0"]
    p1pp_0 = s["p1pp_0"]; p2pp_0 = s["p2pp_0"]
    Lrpp_0 = s["Lrpp_0"]

    # --- Read second-order EOM partials from scratch ---
    q1p2_nu = s["q1p2_nu"]; q1p2_Lr = s["q1p2_Lr"]; q1p2_q1 = s["q1p2_q1"]
    q1p2_q2 = s["q1p2_q2"]; q1p2_p1 = s["q1p2_p1"]; q1p2_p2 = s["q1p2_p2"]
    q2p2_nu = s["q2p2_nu"]; q2p2_Lr = s["q2p2_Lr"]; q2p2_q1 = s["q2p2_q1"]
    q2p2_q2 = s["q2p2_q2"]; q2p2_p1 = s["q2p2_p1"]; q2p2_p2 = s["q2p2_p2"]
    p1p2_nu = s["p1p2_nu"]; p1p2_Lr = s["p1p2_Lr"]; p1p2_q1 = s["p1p2_q1"]
    p1p2_q2 = s["p1p2_q2"]; p1p2_p1 = s["p1p2_p1"]; p1p2_p2 = s["p1p2_p2"]
    p2p2_nu = s["p2p2_nu"]; p2p2_Lr = s["p2p2_Lr"]; p2p2_q1 = s["p2p2_q1"]
    p2p2_q2 = s["p2p2_q2"]; p2p2_p1 = s["p2p2_p1"]; p2p2_p2 = s["p2p2_p2"]
    Lrp2_nu = s["Lrp2_nu"]; Lrp2_Lr = s["Lrp2_Lr"]; Lrp2_q1 = s["Lrp2_q1"]
    Lrp2_q2 = s["Lrp2_q2"]; Lrp2_p1 = s["Lrp2_p1"]; Lrp2_p2 = s["Lrp2_p2"]

    # --- Read STM partial accumulators (written by evaluate_order_1) ---
    Lr_nu = s["Lr_nu"]; Lr_Lr = s["Lr_Lr"]; Lr_q1 = s["Lr_q1"]
    Lr_q2 = s["Lr_q2"]; Lr_p1 = s["Lr_p1"]; Lr_p2 = s["Lr_p2"]
    q1_nu = s["q1_nu"]; q1_Lr = s["q1_Lr"]; q1_q1 = s["q1_q1"]
    q1_q2 = s["q1_q2"]; q1_p1 = s["q1_p1"]; q1_p2 = s["q1_p2"]
    q2_nu = s["q2_nu"]; q2_Lr = s["q2_Lr"]; q2_q1 = s["q2_q1"]
    q2_q2 = s["q2_q2"]; q2_p1 = s["q2_p1"]; q2_p2 = s["q2_p2"]
    p1_nu = s["p1_nu"]; p1_Lr = s["p1_Lr"]; p1_q1 = s["p1_q1"]
    p1_q2 = s["p1_q2"]; p1_p1 = s["p1_p1"]; p1_p2 = s["p1_p2"]
    p2_nu = s["p2_nu"]; p2_Lr = s["p2_Lr"]; p2_q1 = s["p2_q1"]
    p2_q2 = s["p2_q2"]; p2_p1 = s["p2_p1"]; p2_p2 = s["p2_p2"]

    # --- Polynomial evaluation (order-2 terms) ---
    dt2 = dt_norm**2

    ctx.y_prop[:, 1]   += q1pp_0 * dt2 / 2
    ctx.y_prop[:, 2]   += q2pp_0 * dt2 / 2
    ctx.y_prop[:, 3]   += p1pp_0 * dt2 / 2
    ctx.y_prop[:, 4]   += p2pp_0 * dt2 / 2
    ctx.y_prop[:, 5]   += Lrpp_0 * dt2 / 2

    # --- STM accumulator updates (order-2 terms) ---
    # Lr derivatives
    Lr_nu = Lr_nu       + Lrp2_nu*dt2/2
    Lr_Lr = Lr_Lr       + Lrp2_Lr*dt2/2
    Lr_q1 = Lr_q1       + Lrp2_q1*dt2/2
    Lr_q2 = Lr_q2       + Lrp2_q2*dt2/2
    Lr_p1 = Lr_p1       + Lrp2_p1*dt2/2
    Lr_p2 = Lr_p2       + Lrp2_p2*dt2/2
    # q1 derivatives
    q1_nu = q1_nu       + q1p2_nu*dt2/2
    q1_Lr = q1_Lr       + q1p2_Lr*dt2/2
    q1_q1 = q1_q1       + q1p2_q1*dt2/2
    q1_q2 = q1_q2       + q1p2_q2*dt2/2
    q1_p1 = q1_p1       + q1p2_p1*dt2/2
    q1_p2 = q1_p2       + q1p2_p2*dt2/2
    # q2 derivatives
    q2_nu = q2_nu       + q2p2_nu*dt2/2
    q2_Lr = q2_Lr       + q2p2_Lr*dt2/2
    q2_q1 = q2_q1       + q2p2_q1*dt2/2
    q2_q2 = q2_q2       + q2p2_q2*dt2/2
    q2_p1 = q2_p1       + q2p2_p1*dt2/2
    q2_p2 = q2_p2       + q2p2_p2*dt2/2
    # p1 derivatives
    p1_nu = p1_nu       + p1p2_nu*dt2/2
    p1_Lr = p1_Lr       + p1p2_Lr*dt2/2
    p1_q1 = p1_q1       + p1p2_q1*dt2/2
    p1_q2 = p1_q2       + p1p2_q2*dt2/2
    p1_p1 = p1_p1       + p1p2_p1*dt2/2
    p1_p2 = p1_p2       + p1p2_p2*dt2/2
    # p2 derivatives
    p2_nu = p2_nu       + p2p2_nu*dt2/2
    p2_Lr = p2_Lr       + p2p2_Lr*dt2/2
    p2_q1 = p2_q1       + p2p2_q1*dt2/2
    p2_q2 = p2_q2       + p2p2_q2*dt2/2
    p2_p1 = p2_p1       + p2p2_p1*dt2/2
    p2_p2 = p2_p2       + p2p2_p2*dt2/2

    # --- Store updated STM partial accumulators ---
    s["Lr_nu"] = Lr_nu; s["Lr_Lr"] = Lr_Lr; s["Lr_q1"] = Lr_q1
    s["Lr_q2"] = Lr_q2; s["Lr_p1"] = Lr_p1; s["Lr_p2"] = Lr_p2
    s["q1_nu"] = q1_nu; s["q1_Lr"] = q1_Lr; s["q1_q1"] = q1_q1
    s["q1_q2"] = q1_q2; s["q1_p1"] = q1_p1; s["q1_p2"] = q1_p2
    s["q2_nu"] = q2_nu; s["q2_Lr"] = q2_Lr; s["q2_q1"] = q2_q1
    s["q2_q2"] = q2_q2; s["q2_p1"] = q2_p1; s["q2_p2"] = q2_p2
    s["p1_nu"] = p1_nu; s["p1_Lr"] = p1_Lr; s["p1_q1"] = p1_q1
    s["p1_q2"] = p1_q2; s["p1_p1"] = p1_p1; s["p1_p2"] = p1_p2
    s["p2_nu"] = p2_nu; s["p2_Lr"] = p2_Lr; s["p2_q1"] = p2_q1
    s["p2_q2"] = p2_q2; s["p2_p1"] = p2_p1; s["p2_p2"] = p2_p2


def compute_order_2(ctx: GEqOEPropagationContext) -> None:  # noqa: C901
    """Compute the order-2 Taylor expansion and populate *ctx*.

    Thin wrapper that calls :func:`compute_coefficients_2` followed by
    :func:`evaluate_order_2`.  Preserves the original API.
    """
    compute_coefficients_2(ctx)
    evaluate_order_2(ctx)
