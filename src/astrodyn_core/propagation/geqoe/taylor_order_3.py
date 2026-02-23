"""Taylor series order-3 computation for J2-perturbed GEqOE propagator."""

from __future__ import annotations

import numpy as np

from astrodyn_core.propagation.geqoe._math_compat import (
    derivatives_of_inverse,
    derivatives_of_inverse_wrt_param,
    derivatives_of_product,
    derivatives_of_product_wrt_param,
)
from astrodyn_core.propagation.geqoe.state import GEqOEPropagationContext


def compute_order_3(ctx: GEqOEPropagationContext) -> None:  # noqa: C901
    """Compute order-2 first, then add order-3 Taylor terms.

    Faithfully extracts legacy propagator.py lines 1220-1746
    (the ``if order > 2:`` block).
    """
    from astrodyn_core.propagation.geqoe.taylor_order_2 import compute_order_2
    compute_order_2(ctx)

    s = ctx.scratch
    dt_norm = ctx.dt_norm

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
    fih = s["fih"]; fir = s["fir"]
    fir2 = s["fir2"]; fir3 = s["fir3"]
    fihr3 = s["fihr3"]; fiD = s["fiD"]
    c = s["c"]; h = s["h"]; hr = s["hr"]
    d = s["d"]; wh = s["wh"]; I = s["I"]; U = s["U"]
    delta = s["delta"]; zg = s["zg"]
    GAMMA_ = s["GAMMA_"]
    xi1 = s["xi1"]; xi2 = s["xi2"]
    fUz = s["fUz"]
    qs = s["qs"]
    C = s["C"]; D = s["D"]
    rpn = s["rpn"]

    # EOM values from order 1
    p1p_0 = s["p1p_0"]; p2p_0 = s["p2p_0"]
    q1p_0 = s["q1p_0"]; q2p_0 = s["q2p_0"]

    # EOM values from order 2
    p1pp_0 = s["p1pp_0"]; p2pp_0 = s["p2pp_0"]
    q1pp_0 = s["q1pp_0"]; q2pp_0 = s["q2pp_0"]

    # Order-2 intermediates
    rpp = s["rpp"]
    cosLp = s["cosLp"]; sinLp = s["sinLp"]
    hp = s["hp"]; dp = s["dp"]; whp = s["whp"]
    Ip = s["Ip"]; Up = s["Up"]
    bp = s["bp"]; cp = s["cp"]
    alphap = s["alphap"]; deltap = s["deltap"]
    zgp = s["zgp"]; fUzp = s["fUzp"]
    GAMMAp_ = s["GAMMAp_"]
    xi1p = s["xi1p"]; xi2p = s["xi2p"]
    qsp = s["qsp"]; psp = s["psp"]
    Cp = s["Cp"]; Dp = s["Dp"]
    rpn_p = s["rpn_p"]
    fiDp = s["fiDp"]
    fibp = s["fibp"]; ficp = s["ficp"]; fihp = s["fihp"]
    fialphap = s["fialphap"]
    fibm1_p = s["fibm1_p"]
    fihr3p = s["fihr3p"]
    firp = s["firp"]; firp2 = s["firp2"]
    fir2p = s["fir2p"]; fir2p2 = s["fir2p2"]
    fir3p = s["fir3p"]; fir3p2 = s["fir3p2"]
    f2rp = s["f2rp"]; f2rpp = s["f2rpp"]
    f2cp = s["f2cp"]
    r3p2 = s["r3p2"]
    hr3p = s["hr3p"]

    # Vectors
    r_vector = s["r_vector"]; r2_vector = s["r2_vector"]; r3_vector = s["r3_vector"]
    beta_vector = s["beta_vector"]; c_vector = s["c_vector"]
    h_vector = s["h_vector"]; bm1_vector = s["bm1_vector"]
    alpha_vector = s["alpha_vector"]; Delta_vector = s["Delta_vector"]
    hr3_vector = s["hr3_vector"]

    # r partials
    r_nu = s["r_nu"]; r_Lr = s["r_Lr"]; r_q1 = s["r_q1"]; r_q2 = s["r_q2"]
    r_p1 = s["r_p1"]; r_p2 = s["r_p2"]
    rp_nu = s["rp_nu"]; rp_Lr = s["rp_Lr"]; rp_q1 = s["rp_q1"]; rp_q2 = s["rp_q2"]
    rp_p1 = s["rp_p1"]; rp_p2 = s["rp_p2"]
    rpp_nu = s["rpp_nu"]; rpp_Lr = s["rpp_Lr"]; rpp_q1 = s["rpp_q1"]
    rpp_q2 = s["rpp_q2"]; rpp_p1 = s["rpp_p1"]; rpp_p2 = s["rpp_p2"]
    r2_nu = s["r2_nu"]; r2_Lr = s["r2_Lr"]; r2_q1 = s["r2_q1"]; r2_q2 = s["r2_q2"]
    r2_p1 = s["r2_p1"]; r2_p2 = s["r2_p2"]
    r3_nu = s["r3_nu"]; r3_Lr = s["r3_Lr"]; r3_q1 = s["r3_q1"]; r3_q2 = s["r3_q2"]
    r3_p1 = s["r3_p1"]; r3_p2 = s["r3_p2"]

    # beta/alpha partials
    beta_nu = s["beta_nu"]; beta_Lr = s["beta_Lr"]; beta_q1 = s["beta_q1"]
    beta_q2 = s["beta_q2"]; beta_p1 = s["beta_p1"]; beta_p2 = s["beta_p2"]
    alpha_nu = s["alpha_nu"]; alpha_Lr = s["alpha_Lr"]; alpha_q1 = s["alpha_q1"]
    alpha_q2 = s["alpha_q2"]; alpha_p1 = s["alpha_p1"]; alpha_p2 = s["alpha_p2"]

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

    # EOM partials from order 1
    q1p_nu = s["q1p_nu"]; q1p_Lr = s["q1p_Lr"]; q1p_q1 = s["q1p_q1"]
    q1p_q2 = s["q1p_q2"]; q1p_p1 = s["q1p_p1"]; q1p_p2 = s["q1p_p2"]
    q2p_nu = s["q2p_nu"]; q2p_Lr = s["q2p_Lr"]; q2p_q1 = s["q2p_q1"]
    q2p_q2 = s["q2p_q2"]; q2p_p1 = s["q2p_p1"]; q2p_p2 = s["q2p_p2"]
    p1p_nu = s["p1p_nu"]; p1p_Lr = s["p1p_Lr"]; p1p_q1 = s["p1p_q1"]
    p1p_q2 = s["p1p_q2"]; p1p_p1 = s["p1p_p1"]; p1p_p2 = s["p1p_p2"]
    p2p_nu = s["p2p_nu"]; p2p_Lr = s["p2p_Lr"]; p2p_q1 = s["p2p_q1"]
    p2p_q2 = s["p2p_q2"]; p2p_p1 = s["p2p_p1"]; p2p_p2 = s["p2p_p2"]

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

    # Higher-order inverse derivative values (from order 1)
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

    # hr3 partials (order 1)
    hr3_nu = s["hr3_nu"]; hr3_Lr = s["hr3_Lr"]; hr3_q1 = s["hr3_q1"]
    hr3_q2 = s["hr3_q2"]; hr3_p1 = s["hr3_p1"]; hr3_p2 = s["hr3_p2"]
    fihr3_nu = s["fihr3_nu"]; fihr3_Lr = s["fihr3_Lr"]
    fihr3_q1 = s["fihr3_q1"]; fihr3_q2 = s["fihr3_q2"]
    fihr3_p1 = s["fihr3_p1"]; fihr3_p2 = s["fihr3_p2"]

    # Order-2 partials of intermediates
    Xp_nu = s["Xp_nu"]; Xp_Lr = s["Xp_Lr"]; Xp_q1 = s["Xp_q1"]
    Xp_q2 = s["Xp_q2"]; Xp_p1 = s["Xp_p1"]; Xp_p2 = s["Xp_p2"]
    Yp_nu = s["Yp_nu"]; Yp_Lr = s["Yp_Lr"]; Yp_q1 = s["Yp_q1"]
    Yp_q2 = s["Yp_q2"]; Yp_p1 = s["Yp_p1"]; Yp_p2 = s["Yp_p2"]

    cosLp_nu = s["cosLp_nu"]; cosLp_Lr = s["cosLp_Lr"]; cosLp_q1 = s["cosLp_q1"]
    cosLp_q2 = s["cosLp_q2"]; cosLp_p1 = s["cosLp_p1"]; cosLp_p2 = s["cosLp_p2"]
    sinLp_nu = s["sinLp_nu"]; sinLp_Lr = s["sinLp_Lr"]; sinLp_q1 = s["sinLp_q1"]
    sinLp_q2 = s["sinLp_q2"]; sinLp_p1 = s["sinLp_p1"]; sinLp_p2 = s["sinLp_p2"]

    zgp_nu = s["zgp_nu"]; zgp_Lr = s["zgp_Lr"]; zgp_q1 = s["zgp_q1"]
    zgp_q2 = s["zgp_q2"]; zgp_p1 = s["zgp_p1"]; zgp_p2 = s["zgp_p2"]

    fUzp_nu = s["fUzp_nu"]; fUzp_Lr = s["fUzp_Lr"]; fUzp_q1 = s["fUzp_q1"]
    fUzp_q2 = s["fUzp_q2"]; fUzp_p1 = s["fUzp_p1"]; fUzp_p2 = s["fUzp_p2"]

    Up_nu = s["Up_nu"]; Up_Lr = s["Up_Lr"]; Up_q1 = s["Up_q1"]
    Up_q2 = s["Up_q2"]; Up_p1 = s["Up_p1"]; Up_p2 = s["Up_p2"]

    bp_nu = s["bp_nu"]; bp_Lr = s["bp_Lr"]; bp_q1 = s["bp_q1"]
    bp_q2 = s["bp_q2"]; bp_p1 = s["bp_p1"]; bp_p2 = s["bp_p2"]

    cp_nu = s["cp_nu"]; cp_Lr = s["cp_Lr"]; cp_q1 = s["cp_q1"]
    cp_q2 = s["cp_q2"]; cp_p1 = s["cp_p1"]; cp_p2 = s["cp_p2"]

    hp_nu = s["hp_nu"]; hp_Lr = s["hp_Lr"]; hp_q1 = s["hp_q1"]
    hp_q2 = s["hp_q2"]; hp_p1 = s["hp_p1"]; hp_p2 = s["hp_p2"]

    alphap_nu = s["alphap_nu"]; alphap_Lr = s["alphap_Lr"]
    alphap_q1 = s["alphap_q1"]; alphap_q2 = s["alphap_q2"]
    alphap_p1 = s["alphap_p1"]; alphap_p2 = s["alphap_p2"]

    deltap_nu = s["deltap_nu"]; deltap_Lr = s["deltap_Lr"]
    deltap_q1 = s["deltap_q1"]; deltap_q2 = s["deltap_q2"]
    deltap_p1 = s["deltap_p1"]; deltap_p2 = s["deltap_p2"]

    Ip_nu = s["Ip_nu"]; Ip_Lr = s["Ip_Lr"]; Ip_q1 = s["Ip_q1"]
    Ip_q2 = s["Ip_q2"]; Ip_p1 = s["Ip_p1"]; Ip_p2 = s["Ip_p2"]

    dp_nu = s["dp_nu"]; dp_Lr = s["dp_Lr"]; dp_q1 = s["dp_q1"]
    dp_q2 = s["dp_q2"]; dp_p1 = s["dp_p1"]; dp_p2 = s["dp_p2"]

    whp_nu = s["whp_nu"]; whp_Lr = s["whp_Lr"]; whp_q1 = s["whp_q1"]
    whp_q2 = s["whp_q2"]; whp_p1 = s["whp_p1"]; whp_p2 = s["whp_p2"]

    GAMMAp_nu = s["GAMMAp_nu"]; GAMMAp_Lr = s["GAMMAp_Lr"]
    GAMMAp_q1 = s["GAMMAp_q1"]; GAMMAp_q2 = s["GAMMAp_q2"]
    GAMMAp_p1 = s["GAMMAp_p1"]; GAMMAp_p2 = s["GAMMAp_p2"]

    xi1p_nu = s["xi1p_nu"]; xi1p_Lr = s["xi1p_Lr"]; xi1p_q1 = s["xi1p_q1"]
    xi1p_q2 = s["xi1p_q2"]; xi1p_p1 = s["xi1p_p1"]; xi1p_p2 = s["xi1p_p2"]
    xi2p_nu = s["xi2p_nu"]; xi2p_Lr = s["xi2p_Lr"]; xi2p_q1 = s["xi2p_q1"]
    xi2p_q2 = s["xi2p_q2"]; xi2p_p1 = s["xi2p_p1"]; xi2p_p2 = s["xi2p_p2"]

    qsp_nu = s["qsp_nu"]; qsp_Lr = s["qsp_Lr"]; qsp_q1 = s["qsp_q1"]
    qsp_q2 = s["qsp_q2"]; qsp_p1 = s["qsp_p1"]; qsp_p2 = s["qsp_p2"]

    psp_nu = s["psp_nu"]; psp_Lr = s["psp_Lr"]; psp_q1 = s["psp_q1"]
    psp_q2 = s["psp_q2"]; psp_p1 = s["psp_p1"]; psp_p2 = s["psp_p2"]

    Cp_nu = s["Cp_nu"]; Cp_Lr = s["Cp_Lr"]; Cp_q1 = s["Cp_q1"]
    Cp_q2 = s["Cp_q2"]; Cp_p1 = s["Cp_p1"]; Cp_p2 = s["Cp_p2"]

    Dp_nu = s["Dp_nu"]; Dp_Lr = s["Dp_Lr"]; Dp_q1 = s["Dp_q1"]
    Dp_q2 = s["Dp_q2"]; Dp_p1 = s["Dp_p1"]; Dp_p2 = s["Dp_p2"]

    fiDp_nu = s["fiDp_nu"]; fiDp_Lr = s["fiDp_Lr"]; fiDp_q1 = s["fiDp_q1"]
    fiDp_q2 = s["fiDp_q2"]; fiDp_p1 = s["fiDp_p1"]; fiDp_p2 = s["fiDp_p2"]

    rpn_p_nu = s["rpn_p_nu"]; rpn_p_Lr = s["rpn_p_Lr"]; rpn_p_q1 = s["rpn_p_q1"]
    rpn_p_q2 = s["rpn_p_q2"]; rpn_p_p1 = s["rpn_p_p1"]; rpn_p_p2 = s["rpn_p_p2"]

    hr3p_nu = s["hr3p_nu"]; hr3p_Lr = s["hr3p_Lr"]; hr3p_q1 = s["hr3p_q1"]
    hr3p_q2 = s["hr3p_q2"]; hr3p_p1 = s["hr3p_p1"]; hr3p_p2 = s["hr3p_p2"]

    fihr3p_nu = s["fihr3p_nu"]; fihr3p_Lr = s["fihr3p_Lr"]
    fihr3p_q1 = s["fihr3p_q1"]; fihr3p_q2 = s["fihr3p_q2"]
    fihr3p_p1 = s["fihr3p_p1"]; fihr3p_p2 = s["fihr3p_p2"]

    firp2_nu = s["firp2_nu"]; firp2_Lr = s["firp2_Lr"]; firp2_q1 = s["firp2_q1"]
    firp2_q2 = s["firp2_q2"]; firp2_p1 = s["firp2_p1"]; firp2_p2 = s["firp2_p2"]

    fir2p2_nu = s["fir2p2_nu"]; fir2p2_Lr = s["fir2p2_Lr"]; fir2p2_q1 = s["fir2p2_q1"]
    fir2p2_q2 = s["fir2p2_q2"]; fir2p2_p1 = s["fir2p2_p1"]; fir2p2_p2 = s["fir2p2_p2"]

    fir3p2_nu = s["fir3p2_nu"]; fir3p2_Lr = s["fir3p2_Lr"]; fir3p2_q1 = s["fir3p2_q1"]
    fir3p2_q2 = s["fir3p2_q2"]; fir3p2_p1 = s["fir3p2_p1"]; fir3p2_p2 = s["fir3p2_p2"]

    f2rpp_nu = s["f2rpp_nu"]; f2rpp_Lr = s["f2rpp_Lr"]; f2rpp_q1 = s["f2rpp_q1"]
    f2rpp_q2 = s["f2rpp_q2"]; f2rpp_p1 = s["f2rpp_p1"]; f2rpp_p2 = s["f2rpp_p2"]

    f2cp_nu = s["f2cp_nu"]; f2cp_Lr = s["f2cp_Lr"]; f2cp_q1 = s["f2cp_q1"]
    f2cp_q2 = s["f2cp_q2"]; f2cp_p1 = s["f2cp_p1"]; f2cp_p2 = s["f2cp_p2"]

    ficp_nu = s["ficp_nu"]; ficp_Lr = s["ficp_Lr"]; ficp_q1 = s["ficp_q1"]
    ficp_q2 = s["ficp_q2"]; ficp_p1 = s["ficp_p1"]; ficp_p2 = s["ficp_p2"]

    fihp_nu = s["fihp_nu"]; fihp_Lr = s["fihp_Lr"]; fihp_q1 = s["fihp_q1"]
    fihp_q2 = s["fihp_q2"]; fihp_p1 = s["fihp_p1"]; fihp_p2 = s["fihp_p2"]

    fialphap_nu = s["fialphap_nu"]; fialphap_Lr = s["fialphap_Lr"]
    fialphap_q1 = s["fialphap_q1"]; fialphap_q2 = s["fialphap_q2"]
    fialphap_p1 = s["fialphap_p1"]; fialphap_p2 = s["fialphap_p2"]

    fib_nu = s["fib_nu"]; fib_Lr = s["fib_Lr"]; fib_q1 = s["fib_q1"]
    fib_q2 = s["fib_q2"]; fib_p1 = s["fib_p1"]; fib_p2 = s["fib_p2"]

    fibp_nu = s["fibp_nu"]; fibp_Lr = s["fibp_Lr"]; fibp_q1 = s["fibp_q1"]
    fibp_q2 = s["fibp_q2"]; fibp_p1 = s["fibp_p1"]; fibp_p2 = s["fibp_p2"]

    fibm1_p_nu = s["fibm1_p_nu"]; fibm1_p_Lr = s["fibm1_p_Lr"]
    fibm1_p_q1 = s["fibm1_p_q1"]; fibm1_p_q2 = s["fibm1_p_q2"]
    fibm1_p_p1 = s["fibm1_p_p1"]; fibm1_p_p2 = s["fibm1_p_p2"]

    hr_nu = s["hr_nu"]; hr_Lr = s["hr_Lr"]; hr_q1 = s["hr_q1"]
    hr_q2 = s["hr_q2"]; hr_p1 = s["hr_p1"]; hr_p2 = s["hr_p2"]

    qs_nu = s["qs_nu"]; qs_Lr = s["qs_Lr"]; qs_q1 = s["qs_q1"]
    qs_q2 = s["qs_q2"]; qs_p1 = s["qs_p1"]; qs_p2 = s["qs_p2"]

    # Second-order EOM partials
    q1p2_nu = s["q1p2_nu"]; q1p2_Lr = s["q1p2_Lr"]; q1p2_q1 = s["q1p2_q1"]
    q1p2_q2 = s["q1p2_q2"]; q1p2_p1 = s["q1p2_p1"]; q1p2_p2 = s["q1p2_p2"]
    q2p2_nu = s["q2p2_nu"]; q2p2_Lr = s["q2p2_Lr"]; q2p2_q1 = s["q2p2_q1"]
    q2p2_q2 = s["q2p2_q2"]; q2p2_p1 = s["q2p2_p1"]; q2p2_p2 = s["q2p2_p2"]
    p1p2_nu = s["p1p2_nu"]; p1p2_Lr = s["p1p2_Lr"]; p1p2_q1 = s["p1p2_q1"]
    p1p2_q2 = s["p1p2_q2"]; p1p2_p1 = s["p1p2_p1"]; p1p2_p2 = s["p1p2_p2"]
    p2p2_nu = s["p2p2_nu"]; p2p2_Lr = s["p2p2_Lr"]; p2p2_q1 = s["p2p2_q1"]
    p2p2_q2 = s["p2p2_q2"]; p2p2_p1 = s["p2p2_p1"]; p2p2_p2 = s["p2p2_p2"]

    # D_XX_vector  (from order 2's store)
    D_nu_vector = s["D_nu_vector"]; D_Lr_vector = s["D_Lr_vector"]
    D_q1_vector = s["D_q1_vector"]; D_q2_vector = s["D_q2_vector"]
    D_p1_vector = s["D_p1_vector"]; D_p2_vector = s["D_p2_vector"]

    # beta_XX_vector
    beta_nu_vector = s["beta_nu_vector"]; beta_Lr_vector = s["beta_Lr_vector"]
    beta_q1_vector = s["beta_q1_vector"]; beta_q2_vector = s["beta_q2_vector"]
    beta_p1_vector = s["beta_p1_vector"]; beta_p2_vector = s["beta_p2_vector"]

    # c_XX_vector
    c_nu_vector = s["c_nu_vector"]; c_Lr_vector = s["c_Lr_vector"]
    c_q1_vector = s["c_q1_vector"]; c_q2_vector = s["c_q2_vector"]
    c_p1_vector = s["c_p1_vector"]; c_p2_vector = s["c_p2_vector"]

    # h_XX_vector
    h_nu_vector = s["h_nu_vector"]; h_Lr_vector = s["h_Lr_vector"]
    h_q1_vector = s["h_q1_vector"]; h_q2_vector = s["h_q2_vector"]
    h_p1_vector = s["h_p1_vector"]; h_p2_vector = s["h_p2_vector"]

    # alpha_XX_vector
    alpha_nu_vector = s["alpha_nu_vector"]; alpha_Lr_vector = s["alpha_Lr_vector"]
    alpha_q1_vector = s["alpha_q1_vector"]; alpha_q2_vector = s["alpha_q2_vector"]
    alpha_p1_vector = s["alpha_p1_vector"]; alpha_p2_vector = s["alpha_p2_vector"]

    # hr3_XX_vector
    hr3_nu_vector = s["hr3_nu_vector"]; hr3_Lr_vector = s["hr3_Lr_vector"]
    hr3_q1_vector = s["hr3_q1_vector"]; hr3_q2_vector = s["hr3_q2_vector"]
    hr3_p1_vector = s["hr3_p1_vector"]; hr3_p2_vector = s["hr3_p2_vector"]

    # STM partial accumulators
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

    # ================================================================== #
    # THIRD ORDER DERIVATIVES (legacy lines 1220-1746)                    #
    # ================================================================== #

    # # INTERMEDIATE COMPUTATIONS

    qsp2    = 2*(q1p_0**2 + q1_0*q1pp_0 + q2p_0**2 + q2_0*q2pp_0)
    psp2    = 2*(p1p_0**2 + p1_0*p1pp_0 + p2p_0**2 + p2_0*p2pp_0)

    qsp2_nu = 2*(2*q1p_0*q1p_nu+q1_0*q1p2_nu + 2*q2p_0*q2p_nu+q2_0*q2p2_nu)
    qsp2_Lr = 2*(2*q1p_0*q1p_Lr+q1_0*q1p2_Lr + 2*q2p_0*q2p_Lr+q2_0*q2p2_Lr)
    qsp2_q1 = 2*(2*q1p_0*q1p_q1+q1pp_0+q1_0*q1p2_q1 + 2*q2p_0*q2p_q1+q2_0*q2p2_q1)
    qsp2_q2 = 2*(2*q1p_0*q1p_q2+q1_0*q1p2_q2 + 2*q2p_0*q2p_q2+q2pp_0+q2_0*q2p2_q2)
    qsp2_p1 = 2*(2*q1p_0*q1p_p1+q1_0*q1p2_p1 + 2*q2p_0*q2p_p1+q2_0*q2p2_p1)
    qsp2_p2 = 2*(2*q1p_0*q1p_p2+q1_0*q1p2_p2 + 2*q2p_0*q2p_p2+q2_0*q2p2_p2)

    psp2_nu = 2*(2*p1p_0*p1p_nu+p1_0*p1p2_nu + 2*p2p_0*p2p_nu+p2_0*p2p2_nu)
    psp2_Lr = 2*(2*p1p_0*p1p_Lr+p1_0*p1p2_Lr + 2*p2p_0*p2p_Lr+p2_0*p2p2_Lr)
    psp2_q1 = 2*(2*p1p_0*p1p_q1+p1_0*p1p2_q1 + 2*p2p_0*p2p_q1+p2_0*p2p2_q1)
    psp2_q2 = 2*(2*p1p_0*p1p_q2+p1_0*p1p2_q2 + 2*p2p_0*p2p_q2+p2_0*p2p2_q2)
    psp2_p1 = 2*(2*p1p_0*p1p_p1+p1pp_0+p1_0*p1p2_p1 + 2*p2p_0*p2p_p1+p2_0*p2p2_p1)
    psp2_p2 = 2*(2*p1p_0*p1p_p2+p1_0*p1p2_p2 + 2*p2p_0*p2p_p2+p2pp_0+p2_0*p2p2_p2)

    hrp     = hp*fir + h*firp
    Xpp     = rpp*cosL + rp*cosLp - hrp*sinL - hr*sinLp
    Ypp     = rpp*sinL + rp*sinLp + hrp*cosL + hr*cosLp

    hrp_nu = hp_nu*fir+hp*fir_nu + h_nu*firp+h*firp_nu
    hrp_Lr = hp_Lr*fir+hp*fir_Lr + h_Lr*firp+h*firp_Lr
    hrp_q1 = hp_q1*fir+hp*fir_q1 + h_q1*firp+h*firp_q1
    hrp_q2 = hp_q2*fir+hp*fir_q2 + h_q2*firp+h*firp_q2
    hrp_p1 = hp_p1*fir+hp*fir_p1 + h_p1*firp+h*firp_p1
    hrp_p2 = hp_p2*fir+hp*fir_p2 + h_p2*firp+h*firp_p2

    Xpp_nu     = rpp_nu*cosL+rpp*cosL_nu + rp_nu*cosLp+rp*cosLp_nu - hrp_nu*sinL-hrp*sinL_nu - hr_nu*sinLp-hr*sinLp_nu
    Xpp_Lr     = rpp_Lr*cosL+rpp*cosL_Lr + rp_Lr*cosLp+rp*cosLp_Lr - hrp_Lr*sinL-hrp*sinL_Lr - hr_Lr*sinLp-hr*sinLp_Lr
    Xpp_q1     = rpp_q1*cosL+rpp*cosL_q1 + rp_q1*cosLp+rp*cosLp_q1 - hrp_q1*sinL-hrp*sinL_q1 - hr_q1*sinLp-hr*sinLp_q1
    Xpp_q2     = rpp_q2*cosL+rpp*cosL_q2 + rp_q2*cosLp+rp*cosLp_q2 - hrp_q2*sinL-hrp*sinL_q2 - hr_q2*sinLp-hr*sinLp_q2
    Xpp_p1     = rpp_p1*cosL+rpp*cosL_p1 + rp_p1*cosLp+rp*cosLp_p1 - hrp_p1*sinL-hrp*sinL_p1 - hr_p1*sinLp-hr*sinLp_p1
    Xpp_p2     = rpp_p2*cosL+rpp*cosL_p2 + rp_p2*cosLp+rp*cosLp_p2 - hrp_p2*sinL-hrp*sinL_p2 - hr_p2*sinLp-hr*sinLp_p2

    Ypp_nu     = rpp_nu*sinL+rpp*sinL_nu + rp_nu*sinLp+rp*sinLp_nu + hrp_nu*cosL+hrp*cosL_nu + hr_nu*cosLp+hr*cosLp_nu
    Ypp_Lr     = rpp_Lr*sinL+rpp*sinL_Lr + rp_Lr*sinLp+rp*sinLp_Lr + hrp_Lr*cosL+hrp*cosL_Lr + hr_Lr*cosLp+hr*cosLp_Lr
    Ypp_q1     = rpp_q1*sinL+rpp*sinL_q1 + rp_q1*sinLp+rp*sinLp_q1 + hrp_q1*cosL+hrp*cosL_q1 + hr_q1*cosLp+hr*cosLp_q1
    Ypp_q2     = rpp_q2*sinL+rpp*sinL_q2 + rp_q2*sinLp+rp*sinLp_q2 + hrp_q2*cosL+hrp*cosL_q2 + hr_q2*cosLp+hr*cosLp_q2
    Ypp_p1     = rpp_p1*sinL+rpp*sinL_p1 + rp_p1*sinLp+rp*sinLp_p1 + hrp_p1*cosL+hrp*cosL_p1 + hr_p1*cosLp+hr*cosLp_p1
    Ypp_p2     = rpp_p2*sinL+rpp*sinL_p2 + rp_p2*sinLp+rp*sinLp_p2 + hrp_p2*cosL+hrp*cosL_p2 + hr_p2*cosLp+hr*cosLp_p2

    # zg
    Cpp     = 2*(Ypp*q2_0 + 2*Yp*q2p_0 + Y*q2pp_0 - (Xpp*q1_0 + 2*Xp*q1p_0 + X*q1pp_0))
    Dpp     = rpp*qs + 2*rp*qsp + r*qsp2
    Delta_vector = np.append(Delta_vector, Dpp)
    fiDp2   = derivatives_of_inverse(Delta_vector, True)

    zgpp    = Cpp*fiD + 2*Cp*fiDp + C*fiDp2

    Cpp_nu     = 2*(Ypp_nu*q2_0 + 2*(Yp_nu*q2p_0+Yp*q2p_nu) + Y_nu*q2pp_0+Y*q2p2_nu - (Xpp_nu*q1_0 + 2*(Xp_nu*q1p_0+Xp*q1p_nu) + X_nu*q1pp_0+X*q1p2_nu))
    Cpp_Lr     = 2*(Ypp_Lr*q2_0 + 2*(Yp_Lr*q2p_0+Yp*q2p_Lr) + Y_Lr*q2pp_0+Y*q2p2_Lr - (Xpp_Lr*q1_0 + 2*(Xp_Lr*q1p_0+Xp*q1p_Lr) + X_Lr*q1pp_0+X*q1p2_Lr))
    Cpp_q1     = 2*(Ypp_q1*q2_0 + 2*(Yp_q1*q2p_0+Yp*q2p_q1) + Y_q1*q2pp_0+Y*q2p2_q1 - (Xpp+Xpp_q1*q1_0 + 2*(Xp_q1*q1p_0+Xp*q1p_q1) + X_q1*q1pp_0+X*q1p2_q1))
    Cpp_q2     = 2*(Ypp+Ypp_q2*q2_0 + 2*(Yp_q2*q2p_0+Yp*q2p_q2) + Y_q2*q2pp_0+Y*q2p2_q2 - (Xpp_q2*q1_0 + 2*(Xp_q2*q1p_0+Xp*q1p_q2) + X_q2*q1pp_0+X*q1p2_q2))
    Cpp_p1     = 2*(Ypp_p1*q2_0 + 2*(Yp_p1*q2p_0+Yp*q2p_p1) + Y_p1*q2pp_0+Y*q2p2_p1 - (Xpp_p1*q1_0 + 2*(Xp_p1*q1p_0+Xp*q1p_p1) + X_p1*q1pp_0+X*q1p2_p1))
    Cpp_p2     = 2*(Ypp_p2*q2_0 + 2*(Yp_p2*q2p_0+Yp*q2p_p2) + Y_p2*q2pp_0+Y*q2p2_p2 - (Xpp_p2*q1_0 + 2*(Xp_p2*q1p_0+Xp*q1p_p2) + X_p2*q1pp_0+X*q1p2_p2))

    Dpp_nu     = rpp_nu*qs+rpp*qs_nu + 2*(rp_nu*qsp+rp*qsp_nu) + r_nu*qsp2+r*qsp2_nu
    Dpp_Lr     = rpp_Lr*qs+rpp*qs_Lr + 2*(rp_Lr*qsp+rp*qsp_Lr) + r_Lr*qsp2+r*qsp2_Lr
    Dpp_q1     = rpp_q1*qs+rpp*qs_q1 + 2*(rp_q1*qsp+rp*qsp_q1) + r_q1*qsp2+r*qsp2_q1
    Dpp_q2     = rpp_q2*qs+rpp*qs_q2 + 2*(rp_q2*qsp+rp*qsp_q2) + r_q2*qsp2+r*qsp2_q2
    Dpp_p1     = rpp_p1*qs+rpp*qs_p1 + 2*(rp_p1*qsp+rp*qsp_p1) + r_p1*qsp2+r*qsp2_p1
    Dpp_p2     = rpp_p2*qs+rpp*qs_p2 + 2*(rp_p2*qsp+rp*qsp_p2) + r_p2*qsp2+r*qsp2_p2

    D_nu_vector = np.array([D_nu, Dp_nu, Dpp_nu]); D_Lr_vector = np.array([D_Lr, Dp_Lr, Dpp_Lr])
    D_q1_vector = np.array([D_q1, Dp_q1, Dpp_q1]); D_q2_vector = np.array([D_q2, Dp_q2, Dpp_q2])
    D_p1_vector = np.array([D_p1, Dp_p1, Dpp_p1]); D_p2_vector = np.array([D_p2, Dp_p2, Dpp_p2])

    fiDp2_nu    = derivatives_of_inverse_wrt_param(Delta_vector, D_nu_vector, True)
    fiDp2_Lr    = derivatives_of_inverse_wrt_param(Delta_vector, D_Lr_vector, True)
    fiDp2_q1    = derivatives_of_inverse_wrt_param(Delta_vector, D_q1_vector, True)
    fiDp2_q2    = derivatives_of_inverse_wrt_param(Delta_vector, D_q2_vector, True)
    fiDp2_p1    = derivatives_of_inverse_wrt_param(Delta_vector, D_p1_vector, True)
    fiDp2_p2    = derivatives_of_inverse_wrt_param(Delta_vector, D_p2_vector, True)

    zgpp_nu     = Cpp_nu*fiD+Cpp*fiD_nu + 2*(Cp_nu*fiDp+Cp*fiDp_nu) + C_nu*fiDp2+C*fiDp2_nu
    zgpp_Lr     = Cpp_Lr*fiD+Cpp*fiD_Lr + 2*(Cp_Lr*fiDp+Cp*fiDp_Lr) + C_Lr*fiDp2+C*fiDp2_Lr
    zgpp_q1     = Cpp_q1*fiD+Cpp*fiD_q1 + 2*(Cp_q1*fiDp+Cp*fiDp_q1) + C_q1*fiDp2+C*fiDp2_q1
    zgpp_q2     = Cpp_q2*fiD+Cpp*fiD_q2 + 2*(Cp_q2*fiDp+Cp*fiDp_q2) + C_q2*fiDp2+C*fiDp2_q2
    zgpp_p1     = Cpp_p1*fiD+Cpp*fiD_p1 + 2*(Cp_p1*fiDp+Cp*fiDp_p1) + C_p1*fiDp2+C*fiDp2_p1
    zgpp_p2     = Cpp_p2*fiD+Cpp*fiD_p2 + 2*(Cp_p2*fiDp+Cp*fiDp_p2) + C_p2*fiDp2+C*fiDp2_p2

    # U functions
    fUzp2   = -6*(zgp**2+zg*zgpp)
    Upp     = -A*( fUzp2*fir3 + 2*fUzp*fir3p + fUz*fir3p2 )

    fUzp2_nu    = -6*(2*zgp*zgp_nu + zg_nu*zgpp+zg*zgpp_nu)
    fUzp2_Lr    = -6*(2*zgp*zgp_Lr + zg_Lr*zgpp+zg*zgpp_Lr)
    fUzp2_q1    = -6*(2*zgp*zgp_q1 + zg_q1*zgpp+zg*zgpp_q1)
    fUzp2_q2    = -6*(2*zgp*zgp_q2 + zg_q2*zgpp+zg*zgpp_q2)
    fUzp2_p1    = -6*(2*zgp*zgp_p1 + zg_p1*zgpp+zg*zgpp_p1)
    fUzp2_p2    = -6*(2*zgp*zgp_p2 + zg_p2*zgpp+zg*zgpp_p2)

    Upp_nu      = -A*( fUzp2_nu*fir3+fUzp2*fir3_nu + 2*(fUzp_nu*fir3p+fUzp*fir3_p_nu) + fUz_nu*fir3p2+fUz*fir3p2_nu )
    Upp_Lr      = -A*( fUzp2_Lr*fir3+fUzp2*fir3_Lr + 2*(fUzp_Lr*fir3p+fUzp*fir3_p_Lr) + fUz_Lr*fir3p2+fUz*fir3p2_Lr )
    Upp_q1      = -A*( fUzp2_q1*fir3+fUzp2*fir3_q1 + 2*(fUzp_q1*fir3p+fUzp*fir3_p_q1) + fUz_q1*fir3p2+fUz*fir3p2_q1 )
    Upp_q2      = -A*( fUzp2_q2*fir3+fUzp2*fir3_q2 + 2*(fUzp_q2*fir3p+fUzp*fir3_p_q2) + fUz_q2*fir3p2+fUz*fir3p2_q2 )
    Upp_p1      = -A*( fUzp2_p1*fir3+fUzp2*fir3_p1 + 2*(fUzp_p1*fir3p+fUzp*fir3_p_p1) + fUz_p1*fir3p2+fUz*fir3p2_p1 )
    Upp_p2      = -A*( fUzp2_p2*fir3+fUzp2*fir3_p2 + 2*(fUzp_p2*fir3p+fUzp*fir3_p_p2) + fUz_p2*fir3p2+fUz*fir3p2_p2 )

    # beta functions
    bpp     = -1/2*(psp2*fib + psp*fibp)
    beta_vector = np.array([beta, bp, bpp])
    fibp2  = derivatives_of_inverse(beta_vector, True)

    bpp_nu     = -1/2*(psp2_nu*fib+psp2*fib_nu + psp_nu*fibp+psp*fibp_nu)
    bpp_Lr     = -1/2*(psp2_Lr*fib+psp2*fib_Lr + psp_Lr*fibp+psp*fibp_Lr)
    bpp_q1     = -1/2*(psp2_q1*fib+psp2*fib_q1 + psp_q1*fibp+psp*fibp_q1)
    bpp_q2     = -1/2*(psp2_q2*fib+psp2*fib_q2 + psp_q2*fibp+psp*fibp_q2)
    bpp_p1     = -1/2*(psp2_p1*fib+psp2*fib_p1 + psp_p1*fibp+psp*fibp_p1)
    bpp_p2     = -1/2*(psp2_p2*fib+psp2*fib_p2 + psp_p2*fibp+psp*fibp_p2)

    beta_nu_vector = np.array([beta_nu, bp_nu, bpp_nu]); beta_Lr_vector = np.array([beta_Lr, bp_Lr, bpp_Lr])
    beta_q1_vector = np.array([beta_q1, bp_q1, bpp_q1]); beta_q2_vector = np.array([beta_q2, bp_q2, bpp_q2])
    beta_p1_vector = np.array([beta_p1, bp_p1, bpp_p1]); beta_p2_vector = np.array([beta_p2, bp_p2, bpp_p2])

    fibp2_nu  = derivatives_of_inverse_wrt_param(beta_vector, beta_nu_vector, True)
    fibp2_Lr  = derivatives_of_inverse_wrt_param(beta_vector, beta_Lr_vector, True)
    fibp2_q1  = derivatives_of_inverse_wrt_param(beta_vector, beta_q1_vector, True)
    fibp2_q2  = derivatives_of_inverse_wrt_param(beta_vector, beta_q2_vector, True)
    fibp2_p1  = derivatives_of_inverse_wrt_param(beta_vector, beta_p1_vector, True)
    fibp2_p2  = derivatives_of_inverse_wrt_param(beta_vector, beta_p2_vector, True)

    # c functions
    cpp     = (mu_norm**2/nu_0)**(1/3)*bpp
    c_vector = np.append(c_vector, cpp)

    ficp2   = derivatives_of_inverse(c_vector, True)
    f2cp_p  = derivatives_of_product(c_vector, True)

    cpp_nu  = mu_norm**(2/3)*(bpp_nu*nu_0**(-1/3)+bpp*(-1/3)*nu_0**(-4/3))
    cpp_Lr = mu_norm**(2/3)/nu_0**(1/3)*bpp_Lr
    cpp_q1 = mu_norm**(2/3)/nu_0**(1/3)*bpp_q1
    cpp_q2 = mu_norm**(2/3)/nu_0**(1/3)*bpp_q2
    cpp_p1 = mu_norm**(2/3)/nu_0**(1/3)*bpp_p1
    cpp_p2 = mu_norm**(2/3)/nu_0**(1/3)*bpp_p2

    c_nu_vector = np.array([c_nu, cp_nu, cpp_nu]); c_Lr_vector = np.array([c_Lr, cp_Lr, cpp_Lr])
    c_q1_vector = np.array([c_q1, cp_q1, cpp_q1]); c_q2_vector = np.array([c_q2, cp_q2, cpp_q2])
    c_p1_vector = np.array([c_p1, cp_p1, cpp_p1]); c_p2_vector = np.array([c_p2, cp_p2, cpp_p2])

    ficp2_nu  = derivatives_of_inverse_wrt_param(c_vector, c_nu_vector, True)
    ficp2_Lr  = derivatives_of_inverse_wrt_param(c_vector, c_Lr_vector, True)
    ficp2_q1  = derivatives_of_inverse_wrt_param(c_vector, c_q1_vector, True)
    ficp2_q2  = derivatives_of_inverse_wrt_param(c_vector, c_q2_vector, True)
    ficp2_p1  = derivatives_of_inverse_wrt_param(c_vector, c_p1_vector, True)
    ficp2_p2  = derivatives_of_inverse_wrt_param(c_vector, c_p2_vector, True)

    f2cp_p_nu = derivatives_of_product_wrt_param(c_vector, c_nu_vector, True)
    f2cp_p_Lr = derivatives_of_product_wrt_param(c_vector, c_Lr_vector, True)
    f2cp_p_q1 = derivatives_of_product_wrt_param(c_vector, c_q1_vector, True)
    f2cp_p_q2 = derivatives_of_product_wrt_param(c_vector, c_q2_vector, True)
    f2cp_p_p1 = derivatives_of_product_wrt_param(c_vector, c_p1_vector, True)
    f2cp_p_p2 = derivatives_of_product_wrt_param(c_vector, c_p2_vector, True)

    # L functions
    cosLp2   = Xpp*fir + 2*Xp*firp + X*firp2
    sinLp2   = Ypp*fir + 2*Yp*firp + Y*firp2

    cosLp2_nu = Xpp_nu*fir+Xpp*fir_nu + 2*(Xp_nu*firp+Xp*firp_nu) + X_nu*firp2+X*firp2_nu
    cosLp2_Lr = Xpp_Lr*fir+Xpp*fir_Lr + 2*(Xp_Lr*firp+Xp*firp_Lr) + X_Lr*firp2+X*firp2_Lr
    cosLp2_q1 = Xpp_q1*fir+Xpp*fir_q1 + 2*(Xp_q1*firp+Xp*firp_q1) + X_q1*firp2+X*firp2_q1
    cosLp2_q2 = Xpp_q2*fir+Xpp*fir_q2 + 2*(Xp_q2*firp+Xp*firp_q2) + X_q2*firp2+X*firp2_q2
    cosLp2_p1 = Xpp_p1*fir+Xpp*fir_p1 + 2*(Xp_p1*firp+Xp*firp_p1) + X_p1*firp2+X*firp2_p1
    cosLp2_p2 = Xpp_p2*fir+Xpp*fir_p2 + 2*(Xp_p2*firp+Xp*firp_p2) + X_p2*firp2+X*firp2_p2

    sinLp2_nu = Ypp_nu*fir+Ypp*fir_nu + 2*(Yp_nu*firp+Yp*firp_nu) + Y_nu*firp2+Y*firp2_nu
    sinLp2_Lr = Ypp_Lr*fir+Ypp*fir_Lr + 2*(Yp_Lr*firp+Yp*firp_Lr) + Y_Lr*firp2+Y*firp2_Lr
    sinLp2_q1 = Ypp_q1*fir+Ypp*fir_q1 + 2*(Yp_q1*firp+Yp*firp_q1) + Y_q1*firp2+Y*firp2_q1
    sinLp2_q2 = Ypp_q2*fir+Ypp*fir_q2 + 2*(Yp_q2*firp+Yp*firp_q2) + Y_q2*firp2+Y*firp2_q2
    sinLp2_p1 = Ypp_p1*fir+Ypp*fir_p1 + 2*(Yp_p1*firp+Yp*firp_p1) + Y_p1*firp2+Y*firp2_p1
    sinLp2_p2 = Ypp_p2*fir+Ypp*fir_p2 + 2*(Yp_p2*firp+Yp*firp_p2) + Y_p2*firp2+Y*firp2_p2

    # r functions
    rpnp2      = p2pp_0*sinL + 2*p2p_0*sinLp + p2_0*sinLp2 - (p1pp_0*cosL + 2*p1p_0*cosLp + p1_0*cosLp2)
    rp3         = mu_norm * ( rpnp2*fic + 2*rpn_p*ficp + rpn*ficp2 )
    r_vector    = np.append(r_vector, rp3)
    firp3       = derivatives_of_inverse(r_vector, True)
    f2rpp2     = derivatives_of_product(r_vector, True)

    rpnp2_nu   = p2p2_nu*sinL+p2pp_0*sinL_nu + 2*(p2p_nu*sinLp+p2p_0*sinLp_nu) + p2_0*sinLp2_nu - (p1p2_nu*cosL+p1pp_0*cosL_nu + 2*(p1p_nu*cosLp+p1p_0*cosLp_nu) + p1_0*cosLp2_nu)
    rpnp2_Lr   = p2p2_Lr*sinL+p2pp_0*sinL_Lr + 2*(p2p_Lr*sinLp+p2p_0*sinLp_Lr) + p2_0*sinLp2_Lr - (p1p2_Lr*cosL+p1pp_0*cosL_Lr + 2*(p1p_Lr*cosLp+p1p_0*cosLp_Lr) + p1_0*cosLp2_Lr)
    rpnp2_q1   = p2p2_q1*sinL+p2pp_0*sinL_q1 + 2*(p2p_q1*sinLp+p2p_0*sinLp_q1) + p2_0*sinLp2_q1 - (p1p2_q1*cosL+p1pp_0*cosL_q1 + 2*(p1p_q1*cosLp+p1p_0*cosLp_q1) + p1_0*cosLp2_q1)
    rpnp2_q2   = p2p2_q2*sinL+p2pp_0*sinL_q2 + 2*(p2p_q2*sinLp+p2p_0*sinLp_q2) + p2_0*sinLp2_q2 - (p1p2_q2*cosL+p1pp_0*cosL_q2 + 2*(p1p_q2*cosLp+p1p_0*cosLp_q2) + p1_0*cosLp2_q2)
    rpnp2_p1   = p2p2_p1*sinL+p2pp_0*sinL_p1 + 2*(p2p_p1*sinLp+p2p_0*sinLp_p1) + p2_0*sinLp2_p1 - (p1p2_p1*cosL+p1pp_0*cosL_p1 + 2*(p1p_p1*cosLp+p1p_0*cosLp_p1) + cosLp2+p1_0*cosLp2_p1)
    rpnp2_p2   = p2p2_p2*sinL+p2pp_0*sinL_p2 + 2*(p2p_p2*sinLp+p2p_0*sinLp_p2) + sinLp2+p2_0*sinLp2_p2 - (p1p2_p2*cosL+p1pp_0*cosL_p2 + 2*(p1p_p2*cosLp+p1p_0*cosLp_p2) + p1_0*cosLp2_p2)

    rp3_nu      = mu_norm * ( rpnp2_nu*fic+rpnp2*fic_nu + 2*(rpn_p_nu*ficp+rpn_p*ficp_nu) + rpn_nu*ficp2+rpn*ficp2_nu )
    rp3_Lr      = mu_norm * ( rpnp2_Lr*fic+rpnp2*fic_Lr + 2*(rpn_p_Lr*ficp+rpn_p*ficp_Lr) + rpn_Lr*ficp2+rpn*ficp2_Lr )
    rp3_q1      = mu_norm * ( rpnp2_q1*fic+rpnp2*fic_q1 + 2*(rpn_p_q1*ficp+rpn_p*ficp_q1) + rpn_q1*ficp2+rpn*ficp2_q1 )
    rp3_q2      = mu_norm * ( rpnp2_q2*fic+rpnp2*fic_q2 + 2*(rpn_p_q2*ficp+rpn_p*ficp_q2) + rpn_q2*ficp2+rpn*ficp2_q2 )
    rp3_p1      = mu_norm * ( rpnp2_p1*fic+rpnp2*fic_p1 + 2*(rpn_p_p1*ficp+rpn_p*ficp_p1) + rpn_p1*ficp2+rpn*ficp2_p1 )
    rp3_p2      = mu_norm * ( rpnp2_p2*fic+rpnp2*fic_p2 + 2*(rpn_p_p2*ficp+rpn_p*ficp_p2) + rpn_p2*ficp2+rpn*ficp2_p2 )

    r_nu_vector = np.append(r_nu_vector, rp3_nu); r_Lr_vector = np.append(r_Lr_vector, rp3_Lr)
    r_q1_vector = np.append(r_q1_vector, rp3_q1); r_q2_vector = np.append(r_q2_vector, rp3_q2)
    r_p1_vector = np.append(r_p1_vector, rp3_p1); r_p2_vector = np.append(r_p2_vector, rp3_p2)

    firp3_nu   = derivatives_of_inverse_wrt_param(r_vector, r_nu_vector, True)
    firp3_Lr   = derivatives_of_inverse_wrt_param(r_vector, r_Lr_vector, True)
    firp3_q1   = derivatives_of_inverse_wrt_param(r_vector, r_q1_vector, True)
    firp3_q2   = derivatives_of_inverse_wrt_param(r_vector, r_q2_vector, True)
    firp3_p1   = derivatives_of_inverse_wrt_param(r_vector, r_p1_vector, True)
    firp3_p2   = derivatives_of_inverse_wrt_param(r_vector, r_p2_vector, True)

    f2rpp2_nu  = derivatives_of_product_wrt_param(r_vector, r_nu_vector, True)
    f2rpp2_Lr  = derivatives_of_product_wrt_param(r_vector, r_Lr_vector, True)
    f2rpp2_q1  = derivatives_of_product_wrt_param(r_vector, r_q1_vector, True)
    f2rpp2_q2  = derivatives_of_product_wrt_param(r_vector, r_q2_vector, True)
    f2rpp2_p1  = derivatives_of_product_wrt_param(r_vector, r_p1_vector, True)
    f2rpp2_p2  = derivatives_of_product_wrt_param(r_vector, r_p2_vector, True)

    r2_vector   = np.append(r2_vector, 2*f2rpp2)
    fir2p3      = derivatives_of_inverse(r2_vector, True)

    r3p3        = 3*(2*rp**3 + 6*r*rp*rpp + r2*rp3)
    r3_vector   = np.append(r3_vector, r3p3)
    fir3p3     = derivatives_of_inverse(r3_vector, True)

    r3p3_nu = 3*(6*rp**2*rp_nu + 6*(r_nu*rp*rpp+r*(rp_nu*rpp+rp*rpp_nu)) + r2_nu*rp3 + r2*rp3_nu )
    r3p3_Lr = 3*(6*rp**2*rp_Lr + 6*(r_Lr*rp*rpp+r*(rp_Lr*rpp+rp*rpp_Lr)) + r2_Lr*rp3 + r2*rp3_Lr )
    r3p3_q1 = 3*(6*rp**2*rp_q1 + 6*(r_q1*rp*rpp+r*(rp_q1*rpp+rp*rpp_q1)) + r2_q1*rp3 + r2*rp3_q1 )
    r3p3_q2 = 3*(6*rp**2*rp_q2 + 6*(r_q2*rp*rpp+r*(rp_q2*rpp+rp*rpp_q2)) + r2_q2*rp3 + r2*rp3_q2 )
    r3p3_p1 = 3*(6*rp**2*rp_p1 + 6*(r_p1*rp*rpp+r*(rp_p1*rpp+rp*rpp_p1)) + r2_p1*rp3 + r2*rp3_p1 )
    r3p3_p2 = 3*(6*rp**2*rp_p2 + 6*(r_p2*rp*rpp+r*(rp_p2*rpp+rp*rpp_p2)) + r2_p2*rp3 + r2*rp3_p2 )

    r3_nu_vector = np.append(r3_nu_vector, r3p3_nu); r3_Lr_vector = np.append(r3_Lr_vector, r3p3_Lr)
    r3_q1_vector = np.append(r3_q1_vector, r3p3_q1); r3_q2_vector = np.append(r3_q2_vector, r3p3_q2)
    r3_p1_vector = np.append(r3_p1_vector, r3p3_p1); r3_p2_vector = np.append(r3_p2_vector, r3p3_p2)

    fir3_p3_nu  = derivatives_of_inverse_wrt_param(r3_vector, r3_nu_vector, True)
    fir3_p3_Lr  = derivatives_of_inverse_wrt_param(r3_vector, r3_Lr_vector, True)
    fir3_p3_q1  = derivatives_of_inverse_wrt_param(r3_vector, r3_q1_vector, True)
    fir3_p3_q2  = derivatives_of_inverse_wrt_param(r3_vector, r3_q2_vector, True)
    fir3_p3_p1  = derivatives_of_inverse_wrt_param(r3_vector, r3_p1_vector, True)
    fir3_p3_p2  = derivatives_of_inverse_wrt_param(r3_vector, r3_p2_vector, True)

    r2_nu_vector = np.append(r2_nu_vector, 2*f2rpp2_nu); r2_Lr_vector = np.append(r2_Lr_vector, 2*f2rpp2_Lr)
    r2_q1_vector = np.append(r2_q1_vector, 2*f2rpp2_q1); r2_q2_vector = np.append(r2_q2_vector, 2*f2rpp2_q2)
    r2_p1_vector = np.append(r2_p1_vector, 2*f2rpp2_p1); r2_p2_vector = np.append(r2_p2_vector, 2*f2rpp2_p2)

    fir2p3_nu = derivatives_of_inverse_wrt_param(r2_vector, r2_nu_vector, True)
    fir2p3_Lr = derivatives_of_inverse_wrt_param(r2_vector, r2_Lr_vector, True)
    fir2p3_q1 = derivatives_of_inverse_wrt_param(r2_vector, r2_q1_vector, True)
    fir2p3_q2 = derivatives_of_inverse_wrt_param(r2_vector, r2_q2_vector, True)
    fir2p3_p1 = derivatives_of_inverse_wrt_param(r2_vector, r2_p1_vector, True)
    fir2p3_p2 = derivatives_of_inverse_wrt_param(r2_vector, r2_p2_vector, True)

    # h functions
    hpp         = (f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih + (f2cp-2*f2rp*U-r2*Up)*fihp
    h_vector    = np.append(h_vector, hpp)
    fihp2       = derivatives_of_inverse(h_vector, True)

    hpp_nu = (f2cp_p_nu-2*(U_nu*f2rpp+U*f2rpp_nu)-4*(f2rp_nu*Up+f2rp*Up_nu)-r2_nu*Upp-r2*Upp_nu)*fih+(f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih_nu+\
        (f2cp_nu-2*(f2rp_nu*U+f2rp*U_nu)-r2_nu*Up-r2*Up_nu)*fihp+(f2cp-2*f2rp*U-r2*Up)*fihp_nu
    hpp_Lr = (f2cp_p_Lr-2*(U_Lr*f2rpp+U*f2rpp_Lr)-4*(f2rp_Lr*Up+f2rp*Up_Lr)-r2_Lr*Upp-r2*Upp_Lr)*fih+(f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih_Lr+\
        (f2cp_Lr-2*(f2rp_Lr*U+f2rp*U_Lr)-r2_Lr*Up-r2*Up_Lr)*fihp+(f2cp-2*f2rp*U-r2*Up)*fihp_Lr
    hpp_q1 = (f2cp_p_q1-2*(U_q1*f2rpp+U*f2rpp_q1)-4*(f2rp_q1*Up+f2rp*Up_q1)-r2_q1*Upp-r2*Upp_q1)*fih+(f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih_q1+\
        (f2cp_q1-2*(f2rp_q1*U+f2rp*U_q1)-r2_q1*Up-r2*Up_q1)*fihp+(f2cp-2*f2rp*U-r2*Up)*fihp_q1
    hpp_q2 = (f2cp_p_q2-2*(U_q2*f2rpp+U*f2rpp_q2)-4*(f2rp_q2*Up+f2rp*Up_q2)-r2_q2*Upp-r2*Upp_q2)*fih+(f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih_q2+\
        (f2cp_q2-2*(f2rp_q2*U+f2rp*U_q2)-r2_q2*Up-r2*Up_q2)*fihp+(f2cp-2*f2rp*U-r2*Up)*fihp_q2
    hpp_p1 = (f2cp_p_p1-2*(U_p1*f2rpp+U*f2rpp_p1)-4*(f2rp_p1*Up+f2rp*Up_p1)-r2_p1*Upp-r2*Upp_p1)*fih+(f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih_p1+\
        (f2cp_p1-2*(f2rp_p1*U+f2rp*U_p1)-r2_p1*Up-r2*Up_p1)*fihp+(f2cp-2*f2rp*U-r2*Up)*fihp_p1
    hpp_p2 = (f2cp_p_p2-2*(U_p2*f2rpp+U*f2rpp_p2)-4*(f2rp_p2*Up+f2rp*Up_p2)-r2_p2*Upp-r2*Upp_p2)*fih+(f2cp_p-2*U*f2rpp-4*f2rp*Up-r2*Upp)*fih_p2+\
        (f2cp_p2-2*(f2rp_p2*U+f2rp*U_p2)-r2_p2*Up-r2*Up_p2)*fihp+(f2cp-2*f2rp*U-r2*Up)*fihp_p2

    h_nu_vector = np.array([h_nu, hp_nu, hpp_nu]); h_Lr_vector = np.array([h_Lr, hp_Lr, hpp_Lr])
    h_q1_vector = np.array([h_q1, hp_q1, hpp_q1]); h_q2_vector = np.array([h_q2, hp_q2, hpp_q2])
    h_p1_vector = np.array([h_p1, hp_p1, hpp_p1]); h_p2_vector = np.array([h_p2, hp_p2, hpp_p2])

    fihp2_nu = derivatives_of_inverse_wrt_param(h_vector, h_nu_vector, True)
    fihp2_Lr = derivatives_of_inverse_wrt_param(h_vector, h_Lr_vector, True)
    fihp2_q1 = derivatives_of_inverse_wrt_param(h_vector, h_q1_vector, True)
    fihp2_q2 = derivatives_of_inverse_wrt_param(h_vector, h_q2_vector, True)
    fihp2_p1 = derivatives_of_inverse_wrt_param(h_vector, h_p1_vector, True)
    fihp2_p2 = derivatives_of_inverse_wrt_param(h_vector, h_p2_vector, True)

    # beta+1
    bm1_vector  = np.append(bm1_vector, bpp)
    fibm1p2     = derivatives_of_inverse(bm1_vector, True)

    bm1_nu_vector = np.array([beta_nu, bp_nu, bpp_nu])
    bm1_Lr_vector = np.array([beta_Lr, bp_Lr, bpp_Lr])
    bm1_q1_vector = np.array([beta_q1, bp_q1, bpp_q1])
    bm1_q2_vector = np.array([beta_q2, bp_q2, bpp_q2])
    bm1_p1_vector = np.array([beta_p1, bp_p1, bpp_p1])
    bm1_p2_vector = np.array([beta_p2, bp_p2, bpp_p2])

    fibm1_p2_nu     = derivatives_of_inverse_wrt_param(bm1_vector, bm1_nu_vector, True)
    fibm1_p2_Lr     = derivatives_of_inverse_wrt_param(bm1_vector, bm1_Lr_vector, True)
    fibm1_p2_q1     = derivatives_of_inverse_wrt_param(bm1_vector, bm1_q1_vector, True)
    fibm1_p2_q2     = derivatives_of_inverse_wrt_param(bm1_vector, bm1_q2_vector, True)
    fibm1_p2_p1     = derivatives_of_inverse_wrt_param(bm1_vector, bm1_p1_vector, True)
    fibm1_p2_p2     = derivatives_of_inverse_wrt_param(bm1_vector, bm1_p2_vector, True)

    # alpha
    alphap2     = fibm1p2
    alpha_vector = np.append(alpha_vector, alphap2)
    fialphap2   = derivatives_of_inverse(alpha_vector, True)

    alphap2_nu = fibm1_p2_nu; alphap2_Lr = fibm1_p2_Lr
    alphap2_q1 = fibm1_p2_q1; alphap2_q2 = fibm1_p2_q2
    alphap2_p1 = fibm1_p2_p1; alphap2_p2 = fibm1_p2_p2

    alpha_nu_vector = np.array([alpha_nu, alphap_nu, alphap2_nu]); alpha_Lr_vector = np.array([alpha_Lr, alphap_Lr, alphap2_Lr])
    alpha_q1_vector = np.array([alpha_q1, alphap_q1, alphap2_q1]); alpha_q2_vector = np.array([alpha_q2, alphap_q2, alphap2_q2])
    alpha_p1_vector = np.array([alpha_p1, alphap_p1, alphap2_p1]); alpha_p2_vector = np.array([alpha_p2, alphap_p2, alphap2_p2])

    fialphap2_nu    = derivatives_of_inverse_wrt_param(alpha_vector, alpha_nu_vector, True)
    fialphap2_Lr    = derivatives_of_inverse_wrt_param(alpha_vector, alpha_Lr_vector, True)
    fialphap2_q1    = derivatives_of_inverse_wrt_param(alpha_vector, alpha_q1_vector, True)
    fialphap2_q2    = derivatives_of_inverse_wrt_param(alpha_vector, alpha_q2_vector, True)
    fialphap2_p1    = derivatives_of_inverse_wrt_param(alpha_vector, alpha_p1_vector, True)
    fialphap2_p2    = derivatives_of_inverse_wrt_param(alpha_vector, alpha_p2_vector, True)

    # delta
    deltap2     = -qsp2

    deltap2_nu = -qsp2_nu; deltap2_Lr = -qsp2_Lr
    deltap2_q1 = -qsp2_q1; deltap2_q2 = -qsp2_q2
    deltap2_p1 = -qsp2_p1; deltap2_p2 = -qsp2_p2

    # hr3
    hr3p2       = hpp*r3 + 6*hp*r2*rp + 3*h*(2*r*rp**2 + r2*rpp)
    hr3_vector  = np.append(hr3_vector, hr3p2)
    fihr3p2     = derivatives_of_inverse(hr3_vector, True)

    hr3p2_nu = hpp_nu*r3+hpp*r3_nu + 6*(hp_nu*r2*rp+hp*(r2_nu*rp+r2*rp_nu)) + 3*h_nu*(2*r*rp**2 + r2*rpp) + 3*h*(2*(r_nu*rp**2+r*2*rp*rp_nu) + r2_nu*rpp+r2*rpp_nu)
    hr3p2_Lr = hpp_Lr*r3+hpp*r3_Lr + 6*(hp_Lr*r2*rp+hp*(r2_Lr*rp+r2*rp_Lr)) + 3*h_Lr*(2*r*rp**2 + r2*rpp) + 3*h*(2*(r_Lr*rp**2+r*2*rp*rp_Lr) + r2_Lr*rpp+r2*rpp_Lr)
    hr3p2_q1 = hpp_q1*r3+hpp*r3_q1 + 6*(hp_q1*r2*rp+hp*(r2_q1*rp+r2*rp_q1)) + 3*h_q1*(2*r*rp**2 + r2*rpp) + 3*h*(2*(r_q1*rp**2+r*2*rp*rp_q1) + r2_q1*rpp+r2*rpp_q1)
    hr3p2_q2 = hpp_q2*r3+hpp*r3_q2 + 6*(hp_q2*r2*rp+hp*(r2_q2*rp+r2*rp_q2)) + 3*h_q2*(2*r*rp**2 + r2*rpp) + 3*h*(2*(r_q2*rp**2+r*2*rp*rp_q2) + r2_q2*rpp+r2*rpp_q2)
    hr3p2_p1 = hpp_p1*r3+hpp*r3_p1 + 6*(hp_p1*r2*rp+hp*(r2_p1*rp+r2*rp_p1)) + 3*h_p1*(2*r*rp**2 + r2*rpp) + 3*h*(2*(r_p1*rp**2+r*2*rp*rp_p1) + r2_p1*rpp+r2*rpp_p1)
    hr3p2_p2 = hpp_p2*r3+hpp*r3_p2 + 6*(hp_p2*r2*rp+hp*(r2_p2*rp+r2*rp_p2)) + 3*h_p2*(2*r*rp**2 + r2*rpp) + 3*h*(2*(r_p2*rp**2+r*2*rp*rp_p2) + r2_p2*rpp+r2*rpp_p2)

    hr3_nu_vector = np.array([hr3_nu, hr3p_nu, hr3p2_nu]); hr3_Lr_vector = np.array([hr3_Lr, hr3p_Lr, hr3p2_Lr])
    hr3_q1_vector = np.array([hr3_q1, hr3p_q1, hr3p2_q1]); hr3_q2_vector = np.array([hr3_q2, hr3p_q2, hr3p2_q2])
    hr3_p1_vector = np.array([hr3_p1, hr3p_p1, hr3p2_p1]); hr3_p2_vector = np.array([hr3_p2, hr3p_p2, hr3p2_p2])

    fihr3p2_nu    = derivatives_of_inverse_wrt_param(hr3_vector, hr3_nu_vector, True)
    fihr3p2_Lr    = derivatives_of_inverse_wrt_param(hr3_vector, hr3_Lr_vector, True)
    fihr3p2_q1    = derivatives_of_inverse_wrt_param(hr3_vector, hr3_q1_vector, True)
    fihr3p2_q2    = derivatives_of_inverse_wrt_param(hr3_vector, hr3_q2_vector, True)
    fihr3p2_p1    = derivatives_of_inverse_wrt_param(hr3_vector, hr3_p1_vector, True)
    fihr3p2_p2    = derivatives_of_inverse_wrt_param(hr3_vector, hr3_p2_vector, True)

    # GAMMA
    GAMMApp_    = fialphap2 + alphap2*(1-r/a) - 2*alphap*rp/a - alpha*rpp/a
    Ipp         = 3*A*( (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3 + 2*(zgp*delta+zg*deltap)*fihr3p + zg*delta*fihr3p2 )
    dp2         = (hpp-cpp)*fir2 + 2*(hp-cp)*fir2p + (h-c)*fir2p2
    whp2        = Ipp*zg + 2*Ip*zgp + I*zgpp

    xi1p2       = Xpp/a+2*p2pp_0
    xi2p2       = Ypp/a+2*p1pp_0

    GAMMApp_nu  = fialphap2_nu + alphap2_nu*(1-r/a) + alphap2*(-r_nu/a+r/a**2*a_nu) - 2*alphap_nu*rp/a - 2*alphap*(rp_nu/a-rp/a**2*a_nu) - alpha_nu*rpp/a - alpha*(rpp_nu/a-rpp/a**2*a_nu)
    GAMMApp_Lr  = fialphap2_Lr + alphap2_Lr*(1-r/a) - alphap2*r_Lr/a - 2*alphap_Lr*rp/a - 2*alphap*rp_Lr/a - alpha_Lr*rpp/a - alpha*rpp_Lr/a
    GAMMApp_q1  = fialphap2_q1 + alphap2_q1*(1-r/a) - alphap2*r_q1/a - 2*alphap_q1*rp/a - 2*alphap*rp_q1/a - alpha_q1*rpp/a - alpha*rpp_q1/a
    GAMMApp_q2  = fialphap2_q2 + alphap2_q2*(1-r/a) - alphap2*r_q2/a - 2*alphap_q2*rp/a - 2*alphap*rp_q2/a - alpha_q2*rpp/a - alpha*rpp_q2/a
    GAMMApp_p1  = fialphap2_p1 + alphap2_p1*(1-r/a) - alphap2*r_p1/a - 2*alphap_p1*rp/a - 2*alphap*rp_p1/a - alpha_p1*rpp/a - alpha*rpp_p1/a
    GAMMApp_p2  = fialphap2_p2 + alphap2_p2*(1-r/a) - alphap2*r_p2/a - 2*alphap_p2*rp/a - 2*alphap*rp_p2/a - alpha_p2*rpp/a - alpha*rpp_p2/a

    Ipp_nu = 3*A*( (zgpp_nu*delta+zgpp*delta_nu+2*(zgp_nu*deltap+zgp*deltap_nu)+zg_nu*deltap2+zg*deltap2_nu)*fihr3 + (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3_nu + \
                    2*(zgp_nu*delta+zgp*delta_nu+zg_nu*deltap+zg*deltap_nu)*fihr3p + 2*(zgp*delta+zg*deltap)*fihr3p_nu + (zg_nu*delta+zg*delta_nu)*fihr3p2 + zg*delta*fihr3p2_nu)
    Ipp_Lr = 3*A*( (zgpp_Lr*delta+zgpp*delta_Lr+2*(zgp_Lr*deltap+zgp*deltap_Lr)+zg_Lr*deltap2+zg*deltap2_Lr)*fihr3 + (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3_Lr + \
                    2*(zgp_Lr*delta+zgp*delta_Lr+zg_Lr*deltap+zg*deltap_Lr)*fihr3p + 2*(zgp*delta+zg*deltap)*fihr3p_Lr + (zg_Lr*delta+zg*delta_Lr)*fihr3p2 + zg*delta*fihr3p2_Lr)
    Ipp_q1 = 3*A*( (zgpp_q1*delta+zgpp*delta_q1+2*(zgp_q1*deltap+zgp*deltap_q1)+zg_q1*deltap2+zg*deltap2_q1)*fihr3 + (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3_q1 + \
                    2*(zgp_q1*delta+zgp*delta_q1+zg_q1*deltap+zg*deltap_q1)*fihr3p + 2*(zgp*delta+zg*deltap)*fihr3p_q1 + (zg_q1*delta+zg*delta_q1)*fihr3p2 + zg*delta*fihr3p2_q1)
    Ipp_q2 = 3*A*( (zgpp_q2*delta+zgpp*delta_q2+2*(zgp_q2*deltap+zgp*deltap_q2)+zg_q2*deltap2+zg*deltap2_q2)*fihr3 + (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3_q2 + \
                    2*(zgp_q2*delta+zgp*delta_q2+zg_q2*deltap+zg*deltap_q2)*fihr3p + 2*(zgp*delta+zg*deltap)*fihr3p_q2 + (zg_q2*delta+zg*delta_q2)*fihr3p2 + zg*delta*fihr3p2_q2)
    Ipp_p1 = 3*A*( (zgpp_p1*delta+zgpp*delta_p1+2*(zgp_p1*deltap+zgp*deltap_p1)+zg_p1*deltap2+zg*deltap2_p1)*fihr3 + (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3_p1 + \
                    2*(zgp_p1*delta+zgp*delta_p1+zg_p1*deltap+zg*deltap_p1)*fihr3p + 2*(zgp*delta+zg*deltap)*fihr3p_p1 + (zg_p1*delta+zg*delta_p1)*fihr3p2 + zg*delta*fihr3p2_p1)
    Ipp_p2 = 3*A*( (zgpp_p2*delta+zgpp*delta_p2+2*(zgp_p2*deltap+zgp*deltap_p2)+zg_p2*deltap2+zg*deltap2_p2)*fihr3 + (zgpp*delta+2*zgp*deltap+zg*deltap2)*fihr3_p2 + \
                    2*(zgp_p2*delta+zgp*delta_p2+zg_p2*deltap+zg*deltap_p2)*fihr3p + 2*(zgp*delta+zg*deltap)*fihr3p_p2 + (zg_p2*delta+zg*delta_p2)*fihr3p2 + zg*delta*fihr3p2_p2)

    dp2_nu = (hpp_nu-cpp_nu)*fir2+(hpp-cpp)*fir2_nu + 2*(hp_nu-cp_nu)*fir2p + 2*(hp-cp)*fir2p_nu + (h_nu-c_nu)*fir2p2 + (h-c)*fir2p2_nu
    dp2_Lr = (hpp_Lr-cpp_Lr)*fir2+(hpp-cpp)*fir2_Lr + 2*(hp_Lr-cp_Lr)*fir2p + 2*(hp-cp)*fir2p_Lr + (h_Lr-c_Lr)*fir2p2 + (h-c)*fir2p2_Lr
    dp2_q1 = (hpp_q1-cpp_q1)*fir2+(hpp-cpp)*fir2_q1 + 2*(hp_q1-cp_q1)*fir2p + 2*(hp-cp)*fir2p_q1 + (h_q1-c_q1)*fir2p2 + (h-c)*fir2p2_q1
    dp2_q2 = (hpp_q2-cpp_q2)*fir2+(hpp-cpp)*fir2_q2 + 2*(hp_q2-cp_q2)*fir2p + 2*(hp-cp)*fir2p_q2 + (h_q2-c_q2)*fir2p2 + (h-c)*fir2p2_q2
    dp2_p1 = (hpp_p1-cpp_p1)*fir2+(hpp-cpp)*fir2_p1 + 2*(hp_p1-cp_p1)*fir2p + 2*(hp-cp)*fir2p_p1 + (h_p1-c_p1)*fir2p2 + (h-c)*fir2p2_p1
    dp2_p2 = (hpp_p2-cpp_p2)*fir2+(hpp-cpp)*fir2_p2 + 2*(hp_p2-cp_p2)*fir2p + 2*(hp-cp)*fir2p_p2 + (h_p2-c_p2)*fir2p2 + (h-c)*fir2p2_p2

    whp2_nu        = Ipp_nu*zg+Ipp*zg_nu + 2*Ip_nu*zgp + 2*Ip*zgp_nu + I_nu*zgpp + I*zgpp_nu
    whp2_Lr        = Ipp_Lr*zg+Ipp*zg_Lr + 2*Ip_Lr*zgp + 2*Ip*zgp_Lr + I_Lr*zgpp + I*zgpp_Lr
    whp2_q1        = Ipp_q1*zg+Ipp*zg_q1 + 2*Ip_q1*zgp + 2*Ip*zgp_q1 + I_q1*zgpp + I*zgpp_q1
    whp2_q2        = Ipp_q2*zg+Ipp*zg_q2 + 2*Ip_q2*zgp + 2*Ip*zgp_q2 + I_q2*zgpp + I*zgpp_q2
    whp2_p1        = Ipp_p1*zg+Ipp*zg_p1 + 2*Ip_p1*zgp + 2*Ip*zgp_p1 + I_p1*zgpp + I*zgpp_p1
    whp2_p2        = Ipp_p2*zg+Ipp*zg_p2 + 2*Ip_p2*zgp + 2*Ip*zgp_p2 + I_p2*zgpp + I*zgpp_p2

    xi1p2_nu       = Xpp_nu/a-Xpp/a**2*a_nu+2*p2p2_nu; xi2p2_nu = Ypp_nu/a-Ypp/a**2*a_nu+2*p1p2_nu
    xi1p2_Lr       = Xpp_Lr/a+2*p2p2_Lr; xi2p2_Lr       = Ypp_Lr/a+2*p1p2_Lr
    xi1p2_q1       = Xpp_q1/a+2*p2p2_q1; xi2p2_q1       = Ypp_q1/a+2*p1p2_q1
    xi1p2_q2       = Xpp_q2/a+2*p2p2_q2; xi2p2_q2       = Ypp_q2/a+2*p1p2_q2
    xi1p2_p1       = Xpp_p1/a+2*p2p2_p1; xi2p2_p1       = Ypp_p1/a+2*p1p2_p1
    xi1p2_p2       = Xpp_p2/a+2*p2p2_p2; xi2p2_p2       = Ypp_p2/a+2*p1p2_p2

    # ELEMENT p1
    p1ppp_0     = p2pp_0*(d-wh) + 2*p2p_0*(dp-whp) + p2_0*(dp2-whp2) - \
                    U*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 ) - 2*Up*(ficp*xi1+fic*xi1p) - (fic*xi1)*Upp
    # ELEMENT p2
    p2ppp_0     = p1pp_0*(-d+wh) + 2*p1p_0*(-dp+whp) + p1_0*(-dp2+whp2) + \
                    U*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 ) + 2*Up*(ficp*xi2+fic*xi2p) + (fic*xi2)*Upp
    # ELEMENT Lr
    Lrppp_0     = dp2 - whp2  - U*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ ) - \
                    2*Up*(ficp*GAMMA_+fic*GAMMAp_) - Upp*fic*GAMMA_
    # ELEMENT q1
    q1ppp_0     = - Ipp*sinL - 2*Ip*sinLp - I*sinLp2
    # ELEMENT q2
    q2ppp_0     = - Ipp*cosL - 2*Ip*cosLp - I*cosLp2

    dt3 = dt_norm**3

    ctx.y_prop[:, 1]   += q1ppp_0 * dt3 / 6
    ctx.y_prop[:, 2]   += q2ppp_0 * dt3 / 6
    ctx.y_prop[:, 3]   += p1ppp_0 * dt3 / 6
    ctx.y_prop[:, 4]   += p2ppp_0 * dt3 / 6
    ctx.y_prop[:, 5]   += Lrppp_0 * dt3 / 6

    ctx.map_components[:, 2] = [0, q1ppp_0, q2ppp_0, p1ppp_0, p2ppp_0, Lrppp_0]

    # DERIVATIVES WRT INITIAL CONDITIONS (THIRD ORDER)
    # q1p3 derivatives
    q1p3_nu     = -Ipp_nu*sinL-Ipp*sinL_nu -2*Ip_nu*sinLp-2*Ip*sinLp_nu -I_nu*sinLp2-I*sinLp2_nu
    q1p3_Lr     = -Ipp_Lr*sinL-Ipp*sinL_Lr -2*Ip_Lr*sinLp-2*Ip*sinLp_Lr -I_Lr*sinLp2-I*sinLp2_Lr
    q1p3_q1     = -Ipp_q1*sinL-Ipp*sinL_q1 -2*Ip_q1*sinLp-2*Ip*sinLp_q1 -I_q1*sinLp2-I*sinLp2_q1
    q1p3_q2     = -Ipp_q2*sinL-Ipp*sinL_q2 -2*Ip_q2*sinLp-2*Ip*sinLp_q2 -I_q2*sinLp2-I*sinLp2_q2
    q1p3_p1     = -Ipp_p1*sinL-Ipp*sinL_p1 -2*Ip_p1*sinLp-2*Ip*sinLp_p1 -I_p1*sinLp2-I*sinLp2_p1
    q1p3_p2     = -Ipp_p2*sinL-Ipp*sinL_p2 -2*Ip_p2*sinLp-2*Ip*sinLp_p2 -I_p2*sinLp2-I*sinLp2_p2

    # q2p3 derivatives
    q2p3_nu     = -Ipp_nu*cosL-Ipp*cosL_nu -2*Ip_nu*cosLp-2*Ip*cosLp_nu -I_nu*cosLp2-I*cosLp2_nu
    q2p3_Lr     = -Ipp_Lr*cosL-Ipp*cosL_Lr -2*Ip_Lr*cosLp-2*Ip*cosLp_Lr -I_Lr*cosLp2-I*cosLp2_Lr
    q2p3_q1     = -Ipp_q1*cosL-Ipp*cosL_q1 -2*Ip_q1*cosLp-2*Ip*cosLp_q1 -I_q1*cosLp2-I*cosLp2_q1
    q2p3_q2     = -Ipp_q2*cosL-Ipp*cosL_q2 -2*Ip_q2*cosLp-2*Ip*cosLp_q2 -I_q2*cosLp2-I*cosLp2_q2
    q2p3_p1     = -Ipp_p1*cosL-Ipp*cosL_p1 -2*Ip_p1*cosLp-2*Ip*cosLp_p1 -I_p1*cosLp2-I*cosLp2_p1
    q2p3_p2     = -Ipp_p2*cosL-Ipp*cosL_p2 -2*Ip_p2*cosLp-2*Ip*cosLp_p2 -I_p2*cosLp2-I*cosLp2_p2

    # p1p3 derivatives
    p1p3_nu     = p2p2_nu*(d-wh) + p2pp_0*(d_nu-wh_nu) + 2*p2p_nu*(dp-whp) + 2*p2p_0*(dp_nu-whp_nu) + p2_0*(dp2_nu-whp2_nu)-\
        U*( ficp2_nu*xi1+ficp2*xi1_nu + 2*ficp_nu*xi1p + 2*ficp*xi1p_nu + fic_nu*xi1p2 + fic*xi1p2_nu )-U_nu*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 )-\
        2*Up*(ficp_nu*xi1+ficp*xi1_nu+fic_nu*xi1p+fic*xi1p_nu)-2*Up_nu*(ficp*xi1+fic*xi1p) - (fic_nu*xi1+fic*xi1_nu)*Upp  - fic*xi1*Upp_nu
    p1p3_Lr     = p2p2_Lr*(d-wh) + p2pp_0*(d_Lr-wh_Lr) + 2*p2p_Lr*(dp-whp) + 2*p2p_0*(dp_Lr-whp_Lr) + p2_0*(dp2_Lr-whp2_Lr)-\
        U*( ficp2_Lr*xi1+ficp2*xi1_Lr + 2*ficp_Lr*xi1p + 2*ficp*xi1p_Lr + fic_Lr*xi1p2 + fic*xi1p2_Lr )-U_Lr*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 )-\
        2*Up*(ficp_Lr*xi1+ficp*xi1_Lr+fic_Lr*xi1p+fic*xi1p_Lr)-2*Up_Lr*(ficp*xi1+fic*xi1p) - (fic_Lr*xi1+fic*xi1_Lr)*Upp  - fic*xi1*Upp_Lr
    p1p3_q1     = p2p2_q1*(d-wh) + p2pp_0*(d_q1-wh_q1) + 2*p2p_q1*(dp-whp) + 2*p2p_0*(dp_q1-whp_q1) + p2_0*(dp2_q1-whp2_q1)-\
        U*( ficp2_q1*xi1+ficp2*xi1_q1 + 2*ficp_q1*xi1p + 2*ficp*xi1p_q1 + fic_q1*xi1p2 + fic*xi1p2_q1 )-U_q1*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 )-\
        2*Up*(ficp_q1*xi1+ficp*xi1_q1+fic_q1*xi1p+fic*xi1p_q1)-2*Up_q1*(ficp*xi1+fic*xi1p) - (fic_q1*xi1+fic*xi1_q1)*Upp  - fic*xi1*Upp_q1
    p1p3_q2     = p2p2_q2*(d-wh) + p2pp_0*(d_q2-wh_q2) + 2*p2p_q2*(dp-whp) + 2*p2p_0*(dp_q2-whp_q2) + p2_0*(dp2_q2-whp2_q2)-\
        U*( ficp2_q2*xi1+ficp2*xi1_q2 + 2*ficp_q2*xi1p + 2*ficp*xi1p_q2 + fic_q2*xi1p2 + fic*xi1p2_q2 )-U_q2*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 )-\
        2*Up*(ficp_q2*xi1+ficp*xi1_q2+fic_q2*xi1p+fic*xi1p_q2)-2*Up_q2*(ficp*xi1+fic*xi1p) - (fic_q2*xi1+fic*xi1_q2)*Upp  - fic*xi1*Upp_q2
    p1p3_p1     = p2p2_p1*(d-wh) + p2pp_0*(d_p1-wh_p1) + 2*p2p_p1*(dp-whp) + 2*p2p_0*(dp_p1-whp_p1) + p2_0*(dp2_p1-whp2_p1)-\
        U*( ficp2_p1*xi1+ficp2*xi1_p1 + 2*ficp_p1*xi1p + 2*ficp*xi1p_p1 + fic_p1*xi1p2 + fic*xi1p2_p1 )-U_p1*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 )-\
        2*Up*(ficp_p1*xi1+ficp*xi1_p1+fic_p1*xi1p+fic*xi1p_p1)-2*Up_p1*(ficp*xi1+fic*xi1p) - (fic_p1*xi1+fic*xi1_p1)*Upp  - fic*xi1*Upp_p1
    p1p3_p2     = p2p2_p2*(d-wh) + p2pp_0*(d_p2-wh_p2) + 2*p2p_p2*(dp-whp) + 2*p2p_0*(dp_p2-whp_p2) + (dp2-whp2) + p2_0*(dp2_p2-whp2_p2)-\
        U*( ficp2_p2*xi1+ficp2*xi1_p2 + 2*ficp_p2*xi1p + 2*ficp*xi1p_p2 + fic_p2*xi1p2 + fic*xi1p2_p2 )-U_p2*( ficp2*xi1 + 2*ficp*xi1p + fic*xi1p2 )-\
        2*Up*(ficp_p2*xi1+ficp*xi1_p2+fic_p2*xi1p+fic*xi1p_p2)-2*Up_p2*(ficp*xi1+fic*xi1p) - (fic_p2*xi1+fic*xi1_p2)*Upp  - fic*xi1*Upp_p2

    # p2p3 derivatives
    p2p3_nu     = p1p2_nu*(-d+wh) + p1pp_0*(-d_nu+wh_nu) + 2*p1p_nu*(-dp+whp) + 2*p1p_0*(-dp_nu+whp_nu) + p1_0*(-dp2_nu+whp2_nu)+\
        U*( ficp2_nu*xi2+ficp2*xi2_nu + 2*ficp_nu*xi2p + 2*ficp*xi2p_nu + fic_nu*xi2p2 + fic*xi2p2_nu )+U_nu*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 )+\
        2*Up*(ficp_nu*xi2+ficp*xi2_nu+fic_nu*xi2p+fic*xi2p_nu)+2*Up_nu*(ficp*xi2+fic*xi2p) + (fic_nu*xi2+fic*xi2_nu)*Upp  + fic*xi2*Upp_nu
    p2p3_Lr     = p1p2_Lr*(-d+wh) + p1pp_0*(-d_Lr+wh_Lr) + 2*p1p_Lr*(-dp+whp) + 2*p1p_0*(-dp_Lr+whp_Lr) + p1_0*(-dp2_Lr+whp2_Lr)+\
        U*( ficp2_Lr*xi2+ficp2*xi2_Lr + 2*ficp_Lr*xi2p + 2*ficp*xi2p_Lr + fic_Lr*xi2p2 + fic*xi2p2_Lr )+U_Lr*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 )+\
        2*Up*(ficp_Lr*xi2+ficp*xi2_Lr+fic_Lr*xi2p+fic*xi2p_Lr)+2*Up_Lr*(ficp*xi2+fic*xi2p) + (fic_Lr*xi2+fic*xi2_Lr)*Upp  + fic*xi2*Upp_Lr
    p2p3_q1     = p1p2_q1*(-d+wh) + p1pp_0*(-d_q1+wh_q1) + 2*p1p_q1*(-dp+whp) + 2*p1p_0*(-dp_q1+whp_q1) + p1_0*(-dp2_q1+whp2_q1)+\
        U*( ficp2_q1*xi2+ficp2*xi2_q1 + 2*ficp_q1*xi2p + 2*ficp*xi2p_q1 + fic_q1*xi2p2 + fic*xi2p2_q1 )+U_q1*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 )+\
        2*Up*(ficp_q1*xi2+ficp*xi2_q1+fic_q1*xi2p+fic*xi2p_q1)+2*Up_q1*(ficp*xi2+fic*xi2p) + (fic_q1*xi2+fic*xi2_q1)*Upp  + fic*xi2*Upp_q1
    p2p3_q2     = p1p2_q2*(-d+wh) + p1pp_0*(-d_q2+wh_q2) + 2*p1p_q2*(-dp+whp) + 2*p1p_0*(-dp_q2+whp_q2) + p1_0*(-dp2_q2+whp2_q2)+\
        U*( ficp2_q2*xi2+ficp2*xi2_q2 + 2*ficp_q2*xi2p + 2*ficp*xi2p_q2 + fic_q2*xi2p2 + fic*xi2p2_q2 )+U_q2*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 )+\
        2*Up*(ficp_q2*xi2+ficp*xi2_q2+fic_q2*xi2p+fic*xi2p_q2)+2*Up_q2*(ficp*xi2+fic*xi2p) + (fic_q2*xi2+fic*xi2_q2)*Upp  + fic*xi2*Upp_q2
    p2p3_p1     = p1p2_p1*(-d+wh) + p1pp_0*(-d_p1+wh_p1) + 2*p1p_p1*(-dp+whp) + 2*p1p_0*(-dp_p1+whp_p1) + (-dp2+whp2) + p1_0*(-dp2_p1+whp2_p1)+\
        U*( ficp2_p1*xi2+ficp2*xi2_p1 + 2*ficp_p1*xi2p + 2*ficp*xi2p_p1 + fic_p1*xi2p2 + fic*xi2p2_p1 )+U_p1*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 )+\
        2*Up*(ficp_p1*xi2+ficp*xi2_p1+fic_p1*xi2p+fic*xi2p_p1)+2*Up_p1*(ficp*xi2+fic*xi2p) + (fic_p1*xi2+fic*xi2_p1)*Upp  + fic*xi2*Upp_p1
    p2p3_p2     = p1p2_p2*(-d+wh) + p1pp_0*(-d_p2+wh_p2) + 2*p1p_p2*(-dp+whp) + 2*p1p_0*(-dp_p2+whp_p2) + p1_0*(-dp2_p2+whp2_p2)+\
        U*( ficp2_p2*xi2+ficp2*xi2_p2 + 2*ficp_p2*xi2p + 2*ficp*xi2p_p2 + fic_p2*xi2p2 + fic*xi2p2_p2 )+U_p2*( ficp2*xi2 + 2*ficp*xi2p + fic*xi2p2 )+\
        2*Up*(ficp_p2*xi2+ficp*xi2_p2+fic_p2*xi2p+fic*xi2p_p2)+2*Up_p2*(ficp*xi2+fic*xi2p) + (fic_p2*xi2+fic*xi2_p2)*Upp  + fic*xi2*Upp_p2

    # Lrp3 derivatives
    Lrp3_nu = dp2_nu - whp2_nu - U*( ficp2_nu*GAMMA_+ficp2*GAMMA_nu + 2*ficp_nu*GAMMAp_+2*ficp*GAMMAp_nu + fic_nu*GAMMApp_+fic*GAMMApp_nu ) - U_nu*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ )- \
        2*Up*(ficp_nu*GAMMA_+ficp*GAMMA_nu+fic_nu*GAMMAp_+fic*GAMMAp_nu)-2*Up_nu*(ficp*GAMMA_+fic*GAMMAp_) - Upp_nu*fic*GAMMA_ - Upp*(fic*GAMMA_nu+fic_nu*GAMMA_)
    Lrp3_Lr = dp2_Lr - whp2_Lr - U*( ficp2_Lr*GAMMA_+ficp2*GAMMA_Lr + 2*ficp_Lr*GAMMAp_+2*ficp*GAMMAp_Lr + fic_Lr*GAMMApp_+fic*GAMMApp_Lr ) - U_Lr*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ )- \
        2*Up*(ficp_Lr*GAMMA_+ficp*GAMMA_Lr+fic_Lr*GAMMAp_+fic*GAMMAp_Lr)-2*Up_Lr*(ficp*GAMMA_+fic*GAMMAp_) - Upp_Lr*fic*GAMMA_ - Upp*(fic*GAMMA_Lr+fic_Lr*GAMMA_)
    Lrp3_q1 = dp2_q1 - whp2_q1 - U*( ficp2_q1*GAMMA_+ficp2*GAMMA_q1 + 2*ficp_q1*GAMMAp_+2*ficp*GAMMAp_q1 + fic_q1*GAMMApp_+fic*GAMMApp_q1 ) - U_q1*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ )- \
        2*Up*(ficp_q1*GAMMA_+ficp*GAMMA_q1+fic_q1*GAMMAp_+fic*GAMMAp_q1)-2*Up_q1*(ficp*GAMMA_+fic*GAMMAp_) - Upp_q1*fic*GAMMA_ - Upp*(fic*GAMMA_q1+fic_q1*GAMMA_)
    Lrp3_q2 = dp2_q2 - whp2_q2 - U*( ficp2_q2*GAMMA_+ficp2*GAMMA_q2 + 2*ficp_q2*GAMMAp_+2*ficp*GAMMAp_q2 + fic_q2*GAMMApp_+fic*GAMMApp_q2 ) - U_q2*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ )- \
        2*Up*(ficp_q2*GAMMA_+ficp*GAMMA_q2+fic_q2*GAMMAp_+fic*GAMMAp_q2)-2*Up_q2*(ficp*GAMMA_+fic*GAMMAp_) - Upp_q2*fic*GAMMA_ - Upp*(fic*GAMMA_q2+fic_q2*GAMMA_)
    Lrp3_p1 = dp2_p1 - whp2_p1 - U*( ficp2_p1*GAMMA_+ficp2*GAMMA_p1 + 2*ficp_p1*GAMMAp_+2*ficp*GAMMAp_p1 + fic_p1*GAMMApp_+fic*GAMMApp_p1 ) - U_p1*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ )- \
        2*Up*(ficp_p1*GAMMA_+ficp*GAMMA_p1+fic_p1*GAMMAp_+fic*GAMMAp_p1)-2*Up_p1*(ficp*GAMMA_+fic*GAMMAp_) - Upp_p1*fic*GAMMA_ - Upp*(fic*GAMMA_p1+fic_p1*GAMMA_)
    Lrp3_p2 = dp2_p2 - whp2_p2 - U*( ficp2_p2*GAMMA_+ficp2*GAMMA_p2 + 2*ficp_p2*GAMMAp_+2*ficp*GAMMAp_p2 + fic_p2*GAMMApp_+fic*GAMMApp_p2 ) - U_p2*( ficp2*GAMMA_ + 2*ficp*GAMMAp_ + fic*GAMMApp_ )- \
        2*Up*(ficp_p2*GAMMA_+ficp*GAMMA_p2+fic_p2*GAMMAp_+fic*GAMMAp_p2)-2*Up_p2*(ficp*GAMMA_+fic*GAMMAp_) - Upp_p2*fic*GAMMA_ - Upp*(fic*GAMMA_p2+fic_p2*GAMMA_)

    # Lr derivatives
    Lr_nu = Lr_nu       + Lrp3_nu*dt3/6
    Lr_Lr = Lr_Lr       + Lrp3_Lr*dt3/6
    Lr_q1 = Lr_q1       + Lrp3_q1*dt3/6
    Lr_q2 = Lr_q2       + Lrp3_q2*dt3/6
    Lr_p1 = Lr_p1       + Lrp3_p1*dt3/6
    Lr_p2 = Lr_p2       + Lrp3_p2*dt3/6
    # q1 derivatives
    q1_nu = q1_nu       + q1p3_nu*dt3/6
    q1_Lr = q1_Lr       + q1p3_Lr*dt3/6
    q1_q1 = q1_q1       + q1p3_q1*dt3/6
    q1_q2 = q1_q2       + q1p3_q2*dt3/6
    q1_p1 = q1_p1       + q1p3_p1*dt3/6
    q1_p2 = q1_p2       + q1p3_p2*dt3/6
    # q2 derivatives
    q2_nu = q2_nu       + q2p3_nu*dt3/6
    q2_Lr = q2_Lr       + q2p3_Lr*dt3/6
    q2_q1 = q2_q1       + q2p3_q1*dt3/6
    q2_q2 = q2_q2       + q2p3_q2*dt3/6
    q2_p1 = q2_p1       + q2p3_p1*dt3/6
    q2_p2 = q2_p2       + q2p3_p2*dt3/6
    # p1 derivatives
    p1_nu = p1_nu       + p1p3_nu*dt3/6
    p1_Lr = p1_Lr       + p1p3_Lr*dt3/6
    p1_q1 = p1_q1       + p1p3_q1*dt3/6
    p1_q2 = p1_q2       + p1p3_q2*dt3/6
    p1_p1 = p1_p1       + p1p3_p1*dt3/6
    p1_p2 = p1_p2       + p1p3_p2*dt3/6
    # p2 derivatives
    p2_nu = p2_nu       + p2p3_nu*dt3/6
    p2_Lr = p2_Lr       + p2p3_Lr*dt3/6
    p2_q1 = p2_q1       + p2p3_q1*dt3/6
    p2_q2 = p2_q2       + p2p3_q2*dt3/6
    p2_p1 = p2_p1       + p2p3_p1*dt3/6
    p2_p2 = p2_p2       + p2p3_p2*dt3/6

    # ================================================================== #
    # Store everything back into scratch for order 4                      #
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

    # New scalars computed by order 3
    s["Xpp"] = Xpp; s["Ypp"] = Ypp
    s["cosLp2"] = cosLp2; s["sinLp2"] = sinLp2
    s["hpp"] = hpp; s["dp2"] = dp2; s["whp2"] = whp2
    s["Ipp"] = Ipp; s["Upp"] = Upp
    s["bpp"] = bpp; s["cpp"] = cpp
    s["alphap2"] = alphap2; s["deltap2"] = deltap2
    s["zgpp"] = zgpp; s["fUzp2"] = fUzp2
    s["GAMMApp_"] = GAMMApp_
    s["xi1p2"] = xi1p2; s["xi2p2"] = xi2p2
    s["qsp2"] = qsp2; s["psp2"] = psp2
    s["Cpp"] = Cpp; s["Dpp"] = Dpp
    s["rpnp2"] = rpnp2; s["rp3"] = rp3
    s["rp3_nu"] = rp3_nu; s["rp3_Lr"] = rp3_Lr; s["rp3_q1"] = rp3_q1
    s["rp3_q2"] = rp3_q2; s["rp3_p1"] = rp3_p1; s["rp3_p2"] = rp3_p2
    s["fiDp2"] = fiDp2
    s["fibp2"] = fibp2; s["ficp2"] = ficp2; s["fihp2"] = fihp2
    s["fialphap2"] = fialphap2
    s["fibm1p2"] = fibm1p2
    s["fihr3p2"] = fihr3p2
    s["firp3"] = firp3; s["fir2p3"] = fir2p3; s["fir3p3"] = fir3p3
    s["f2rpp2"] = f2rpp2; s["f2cp_p"] = f2cp_p
    s["r3p3"] = r3p3
    s["hr3p2"] = hr3p2
    s["hrp"] = hrp

    # Partials of new order-3 intermediates
    s["Xpp_nu"] = Xpp_nu; s["Xpp_Lr"] = Xpp_Lr; s["Xpp_q1"] = Xpp_q1
    s["Xpp_q2"] = Xpp_q2; s["Xpp_p1"] = Xpp_p1; s["Xpp_p2"] = Xpp_p2
    s["Ypp_nu"] = Ypp_nu; s["Ypp_Lr"] = Ypp_Lr; s["Ypp_q1"] = Ypp_q1
    s["Ypp_q2"] = Ypp_q2; s["Ypp_p1"] = Ypp_p1; s["Ypp_p2"] = Ypp_p2

    s["cosLp2_nu"] = cosLp2_nu; s["cosLp2_Lr"] = cosLp2_Lr; s["cosLp2_q1"] = cosLp2_q1
    s["cosLp2_q2"] = cosLp2_q2; s["cosLp2_p1"] = cosLp2_p1; s["cosLp2_p2"] = cosLp2_p2
    s["sinLp2_nu"] = sinLp2_nu; s["sinLp2_Lr"] = sinLp2_Lr; s["sinLp2_q1"] = sinLp2_q1
    s["sinLp2_q2"] = sinLp2_q2; s["sinLp2_p1"] = sinLp2_p1; s["sinLp2_p2"] = sinLp2_p2

    s["zgpp_nu"] = zgpp_nu; s["zgpp_Lr"] = zgpp_Lr; s["zgpp_q1"] = zgpp_q1
    s["zgpp_q2"] = zgpp_q2; s["zgpp_p1"] = zgpp_p1; s["zgpp_p2"] = zgpp_p2

    s["fUzp2_nu"] = fUzp2_nu; s["fUzp2_Lr"] = fUzp2_Lr; s["fUzp2_q1"] = fUzp2_q1
    s["fUzp2_q2"] = fUzp2_q2; s["fUzp2_p1"] = fUzp2_p1; s["fUzp2_p2"] = fUzp2_p2

    s["Upp_nu"] = Upp_nu; s["Upp_Lr"] = Upp_Lr; s["Upp_q1"] = Upp_q1
    s["Upp_q2"] = Upp_q2; s["Upp_p1"] = Upp_p1; s["Upp_p2"] = Upp_p2

    s["bpp_nu"] = bpp_nu; s["bpp_Lr"] = bpp_Lr; s["bpp_q1"] = bpp_q1
    s["bpp_q2"] = bpp_q2; s["bpp_p1"] = bpp_p1; s["bpp_p2"] = bpp_p2

    s["cpp_nu"] = cpp_nu; s["cpp_Lr"] = cpp_Lr; s["cpp_q1"] = cpp_q1
    s["cpp_q2"] = cpp_q2; s["cpp_p1"] = cpp_p1; s["cpp_p2"] = cpp_p2

    s["hpp_nu"] = hpp_nu; s["hpp_Lr"] = hpp_Lr; s["hpp_q1"] = hpp_q1
    s["hpp_q2"] = hpp_q2; s["hpp_p1"] = hpp_p1; s["hpp_p2"] = hpp_p2

    s["alphap2_nu"] = alphap2_nu; s["alphap2_Lr"] = alphap2_Lr
    s["alphap2_q1"] = alphap2_q1; s["alphap2_q2"] = alphap2_q2
    s["alphap2_p1"] = alphap2_p1; s["alphap2_p2"] = alphap2_p2

    s["deltap2_nu"] = deltap2_nu; s["deltap2_Lr"] = deltap2_Lr
    s["deltap2_q1"] = deltap2_q1; s["deltap2_q2"] = deltap2_q2
    s["deltap2_p1"] = deltap2_p1; s["deltap2_p2"] = deltap2_p2

    s["Ipp_nu"] = Ipp_nu; s["Ipp_Lr"] = Ipp_Lr; s["Ipp_q1"] = Ipp_q1
    s["Ipp_q2"] = Ipp_q2; s["Ipp_p1"] = Ipp_p1; s["Ipp_p2"] = Ipp_p2

    s["dp2_nu"] = dp2_nu; s["dp2_Lr"] = dp2_Lr; s["dp2_q1"] = dp2_q1
    s["dp2_q2"] = dp2_q2; s["dp2_p1"] = dp2_p1; s["dp2_p2"] = dp2_p2

    s["whp2_nu"] = whp2_nu; s["whp2_Lr"] = whp2_Lr; s["whp2_q1"] = whp2_q1
    s["whp2_q2"] = whp2_q2; s["whp2_p1"] = whp2_p1; s["whp2_p2"] = whp2_p2

    s["GAMMApp_nu"] = GAMMApp_nu; s["GAMMApp_Lr"] = GAMMApp_Lr
    s["GAMMApp_q1"] = GAMMApp_q1; s["GAMMApp_q2"] = GAMMApp_q2
    s["GAMMApp_p1"] = GAMMApp_p1; s["GAMMApp_p2"] = GAMMApp_p2

    s["xi1p2_nu"] = xi1p2_nu; s["xi1p2_Lr"] = xi1p2_Lr; s["xi1p2_q1"] = xi1p2_q1
    s["xi1p2_q2"] = xi1p2_q2; s["xi1p2_p1"] = xi1p2_p1; s["xi1p2_p2"] = xi1p2_p2
    s["xi2p2_nu"] = xi2p2_nu; s["xi2p2_Lr"] = xi2p2_Lr; s["xi2p2_q1"] = xi2p2_q1
    s["xi2p2_q2"] = xi2p2_q2; s["xi2p2_p1"] = xi2p2_p1; s["xi2p2_p2"] = xi2p2_p2

    s["qsp2_nu"] = qsp2_nu; s["qsp2_Lr"] = qsp2_Lr; s["qsp2_q1"] = qsp2_q1
    s["qsp2_q2"] = qsp2_q2; s["qsp2_p1"] = qsp2_p1; s["qsp2_p2"] = qsp2_p2

    s["psp2_nu"] = psp2_nu; s["psp2_Lr"] = psp2_Lr; s["psp2_q1"] = psp2_q1
    s["psp2_q2"] = psp2_q2; s["psp2_p1"] = psp2_p1; s["psp2_p2"] = psp2_p2

    s["Cpp_nu"] = Cpp_nu; s["Cpp_Lr"] = Cpp_Lr; s["Cpp_q1"] = Cpp_q1
    s["Cpp_q2"] = Cpp_q2; s["Cpp_p1"] = Cpp_p1; s["Cpp_p2"] = Cpp_p2

    s["Dpp_nu"] = Dpp_nu; s["Dpp_Lr"] = Dpp_Lr; s["Dpp_q1"] = Dpp_q1
    s["Dpp_q2"] = Dpp_q2; s["Dpp_p1"] = Dpp_p1; s["Dpp_p2"] = Dpp_p2

    s["fiDp2_nu"] = fiDp2_nu; s["fiDp2_Lr"] = fiDp2_Lr; s["fiDp2_q1"] = fiDp2_q1
    s["fiDp2_q2"] = fiDp2_q2; s["fiDp2_p1"] = fiDp2_p1; s["fiDp2_p2"] = fiDp2_p2

    s["rpnp2_nu"] = rpnp2_nu; s["rpnp2_Lr"] = rpnp2_Lr; s["rpnp2_q1"] = rpnp2_q1
    s["rpnp2_q2"] = rpnp2_q2; s["rpnp2_p1"] = rpnp2_p1; s["rpnp2_p2"] = rpnp2_p2

    s["rpp_nu"] = rpp_nu; s["rpp_Lr"] = rpp_Lr; s["rpp_q1"] = rpp_q1
    s["rpp_q2"] = rpp_q2; s["rpp_p1"] = rpp_p1; s["rpp_p2"] = rpp_p2

    s["r3p3_nu"] = r3p3_nu; s["r3p3_Lr"] = r3p3_Lr; s["r3p3_q1"] = r3p3_q1
    s["r3p3_q2"] = r3p3_q2; s["r3p3_p1"] = r3p3_p1; s["r3p3_p2"] = r3p3_p2

    s["hr3p2_nu"] = hr3p2_nu; s["hr3p2_Lr"] = hr3p2_Lr; s["hr3p2_q1"] = hr3p2_q1
    s["hr3p2_q2"] = hr3p2_q2; s["hr3p2_p1"] = hr3p2_p1; s["hr3p2_p2"] = hr3p2_p2

    s["fihr3p2_nu"] = fihr3p2_nu; s["fihr3p2_Lr"] = fihr3p2_Lr
    s["fihr3p2_q1"] = fihr3p2_q1; s["fihr3p2_q2"] = fihr3p2_q2
    s["fihr3p2_p1"] = fihr3p2_p1; s["fihr3p2_p2"] = fihr3p2_p2

    s["firp3_nu"] = firp3_nu; s["firp3_Lr"] = firp3_Lr; s["firp3_q1"] = firp3_q1
    s["firp3_q2"] = firp3_q2; s["firp3_p1"] = firp3_p1; s["firp3_p2"] = firp3_p2

    s["fir2p3_nu"] = fir2p3_nu; s["fir2p3_Lr"] = fir2p3_Lr; s["fir2p3_q1"] = fir2p3_q1
    s["fir2p3_q2"] = fir2p3_q2; s["fir2p3_p1"] = fir2p3_p1; s["fir2p3_p2"] = fir2p3_p2

    s["fir3_p3_nu"] = fir3_p3_nu; s["fir3_p3_Lr"] = fir3_p3_Lr; s["fir3_p3_q1"] = fir3_p3_q1
    s["fir3_p3_q2"] = fir3_p3_q2; s["fir3_p3_p1"] = fir3_p3_p1; s["fir3_p3_p2"] = fir3_p3_p2

    s["f2rpp2_nu"] = f2rpp2_nu; s["f2rpp2_Lr"] = f2rpp2_Lr; s["f2rpp2_q1"] = f2rpp2_q1
    s["f2rpp2_q2"] = f2rpp2_q2; s["f2rpp2_p1"] = f2rpp2_p1; s["f2rpp2_p2"] = f2rpp2_p2

    s["f2cp_p_nu"] = f2cp_p_nu; s["f2cp_p_Lr"] = f2cp_p_Lr; s["f2cp_p_q1"] = f2cp_p_q1
    s["f2cp_p_q2"] = f2cp_p_q2; s["f2cp_p_p1"] = f2cp_p_p1; s["f2cp_p_p2"] = f2cp_p_p2

    s["ficp2_nu"] = ficp2_nu; s["ficp2_Lr"] = ficp2_Lr; s["ficp2_q1"] = ficp2_q1
    s["ficp2_q2"] = ficp2_q2; s["ficp2_p1"] = ficp2_p1; s["ficp2_p2"] = ficp2_p2

    s["fihp2_nu"] = fihp2_nu; s["fihp2_Lr"] = fihp2_Lr; s["fihp2_q1"] = fihp2_q1
    s["fihp2_q2"] = fihp2_q2; s["fihp2_p1"] = fihp2_p1; s["fihp2_p2"] = fihp2_p2

    s["fialphap2_nu"] = fialphap2_nu; s["fialphap2_Lr"] = fialphap2_Lr
    s["fialphap2_q1"] = fialphap2_q1; s["fialphap2_q2"] = fialphap2_q2
    s["fialphap2_p1"] = fialphap2_p1; s["fialphap2_p2"] = fialphap2_p2

    s["fibp2_nu"] = fibp2_nu; s["fibp2_Lr"] = fibp2_Lr; s["fibp2_q1"] = fibp2_q1
    s["fibp2_q2"] = fibp2_q2; s["fibp2_p1"] = fibp2_p1; s["fibp2_p2"] = fibp2_p2

    s["fibm1_p2_nu"] = fibm1_p2_nu; s["fibm1_p2_Lr"] = fibm1_p2_Lr
    s["fibm1_p2_q1"] = fibm1_p2_q1; s["fibm1_p2_q2"] = fibm1_p2_q2
    s["fibm1_p2_p1"] = fibm1_p2_p1; s["fibm1_p2_p2"] = fibm1_p2_p2

    s["hrp_nu"] = hrp_nu; s["hrp_Lr"] = hrp_Lr; s["hrp_q1"] = hrp_q1
    s["hrp_q2"] = hrp_q2; s["hrp_p1"] = hrp_p1; s["hrp_p2"] = hrp_p2

    # Updated fic (overwritten by c_vector inverse)
    s["fic"] = fic

    # Third-order EOM partials
    s["q1p3_nu"] = q1p3_nu; s["q1p3_Lr"] = q1p3_Lr; s["q1p3_q1"] = q1p3_q1
    s["q1p3_q2"] = q1p3_q2; s["q1p3_p1"] = q1p3_p1; s["q1p3_p2"] = q1p3_p2
    s["q2p3_nu"] = q2p3_nu; s["q2p3_Lr"] = q2p3_Lr; s["q2p3_q1"] = q2p3_q1
    s["q2p3_q2"] = q2p3_q2; s["q2p3_p1"] = q2p3_p1; s["q2p3_p2"] = q2p3_p2
    s["p1p3_nu"] = p1p3_nu; s["p1p3_Lr"] = p1p3_Lr; s["p1p3_q1"] = p1p3_q1
    s["p1p3_q2"] = p1p3_q2; s["p1p3_p1"] = p1p3_p1; s["p1p3_p2"] = p1p3_p2
    s["p2p3_nu"] = p2p3_nu; s["p2p3_Lr"] = p2p3_Lr; s["p2p3_q1"] = p2p3_q1
    s["p2p3_q2"] = p2p3_q2; s["p2p3_p1"] = p2p3_p1; s["p2p3_p2"] = p2p3_p2
    s["Lrp3_nu"] = Lrp3_nu; s["Lrp3_Lr"] = Lrp3_Lr; s["Lrp3_q1"] = Lrp3_q1
    s["Lrp3_q2"] = Lrp3_q2; s["Lrp3_p1"] = Lrp3_p1; s["Lrp3_p2"] = Lrp3_p2

    # Updated STM partial accumulators
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

    # Third-order EOM values
    s["p1ppp_0"] = p1ppp_0; s["p2ppp_0"] = p2ppp_0; s["Lrppp_0"] = Lrppp_0
    s["q1ppp_0"] = q1ppp_0; s["q2ppp_0"] = q2ppp_0
