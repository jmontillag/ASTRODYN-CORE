"""heyoka cfunc-compiled mean rate evaluator.

Stage C of the performance optimization plan.  Translates the symbolic
mean-rate expressions into a heyoka expression tree and compiles them
to native SIMD machine code via LLVM.  The resulting cfunc replaces the
entire Python-level mean RHS function with a single C function pointer call.

The cfunc takes state variables [p1, p2, q1, q2] and parameters
[nu, s2, s3, s4, s5] and returns [p1_dot, p2_dot, M_dot_total, q1_dot, q2_dot].
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
#  SymPy -> heyoka expression converter
# ---------------------------------------------------------------------------

def _sympy_to_heyoka(expr, var_map):
    """Recursively convert a SymPy expression to a heyoka expression.

    Parameters
    ----------
    expr : sp.Expr
    var_map : dict mapping SymPy symbol name (str) -> heyoka expression
    """
    import heyoka as hy

    if isinstance(expr, sp.Symbol):
        name = expr.name
        if name not in var_map:
            raise ValueError(f"Unknown symbol '{name}' — not in var_map")
        return var_map[name]

    if isinstance(expr, (sp.Integer, sp.Float)):
        return float(expr)

    if isinstance(expr, sp.Rational):
        return float(expr)

    if isinstance(expr, sp.core.numbers.NegativeOne):
        return -1.0

    if isinstance(expr, sp.core.numbers.One):
        return 1.0

    if isinstance(expr, sp.core.numbers.Zero):
        return 0.0

    if isinstance(expr, sp.core.numbers.Half):
        return 0.5

    if isinstance(expr, sp.Add):
        result = _sympy_to_heyoka(expr.args[0], var_map)
        for arg in expr.args[1:]:
            result = result + _sympy_to_heyoka(arg, var_map)
        return result

    if isinstance(expr, sp.Mul):
        result = _sympy_to_heyoka(expr.args[0], var_map)
        for arg in expr.args[1:]:
            result = result * _sympy_to_heyoka(arg, var_map)
        return result

    if isinstance(expr, sp.Pow):
        base_hy = _sympy_to_heyoka(expr.args[0], var_map)
        exp_sp = expr.args[1]
        if isinstance(exp_sp, sp.Integer):
            return base_hy ** int(exp_sp)
        if isinstance(exp_sp, sp.Rational):
            if exp_sp == sp.Rational(1, 2):
                return hy.sqrt(base_hy)
            if exp_sp == sp.Rational(-1, 2):
                return 1.0 / hy.sqrt(base_hy)
        return base_hy ** _sympy_to_heyoka(exp_sp, var_map)

    if isinstance(expr, sp.cos):
        return hy.cos(_sympy_to_heyoka(expr.args[0], var_map))

    if isinstance(expr, sp.sin):
        return hy.sin(_sympy_to_heyoka(expr.args[0], var_map))

    # Handle integer/float atoms that slipped through
    if isinstance(expr, (int, float)):
        return float(expr)

    raise NotImplementedError(
        f"Cannot convert {type(expr).__name__}: {expr!r}"
    )


# ---------------------------------------------------------------------------
#  Load MEAN_DATA from generated_coefficients.py
# ---------------------------------------------------------------------------

def _load_mean_data() -> dict:
    data_file = Path(__file__).resolve().parent / "generated_coefficients.py"
    tree = ast.parse(data_file.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MEAN_DATA"):
            return ast.literal_eval(node.value)
    raise RuntimeError("MEAN_DATA not found")


# ---------------------------------------------------------------------------
#  Build the full mean RHS cfunc
# ---------------------------------------------------------------------------

_CACHED_CFUNC = None
_CACHED_CFUNC_INFO = None


def build_mean_rhs_cfunc():
    """Build a heyoka cfunc for the full mean RHS.

    The cfunc takes:
        vars:  [p1, p2, q1, q2]
        pars:  [nu, s2, s3, s4, s5]

    and returns:
        [p1_dot, p2_dot, M_dot_total, q1_dot, q2_dot]

    Returns (cfunc, info_dict).
    """
    import heyoka as hy

    t_start = time.time()
    mean_data = _load_mean_data()

    # heyoka variables
    p1, p2, q1, q2 = hy.make_vars("p1", "p2", "q1", "q2")

    # Parameters: nu=par[0], s2=par[1], s3=par[2], s4=par[3], s5=par[4]
    nu_hy = hy.par[0]
    s_hy = {2: hy.par[1], 3: hy.par[2], 4: hy.par[3], 5: hy.par[4]}

    # --- Derived orbital quantities ---
    g2 = p1 * p1 + p2 * p2
    g_hy = hy.sqrt(g2)
    Q2 = q1 * q1 + q2 * q2
    Q_hy = hy.sqrt(Q2)

    # Eccentricity parameter q = g / (1 + beta), beta = sqrt(1 - g^2)
    beta_hy = hy.sqrt(1.0 - g2)
    q_ecl = g_hy / (1.0 + beta_hy)

    # cos(omega), sin(omega) from state variables
    # omega = atan2(p1,p2) - atan2(q1,q2)
    # cos(omega) = (p2*q2 + p1*q1) / (g*Q)
    # sin(omega) = (p1*q2 - p2*q1) / (g*Q)
    gQ_inv = 1.0 / (g_hy * Q_hy)
    cos_omega = (p2 * q2 + p1 * q1) * gQ_inv
    sin_omega = (p1 * q2 - p2 * q1) * gQ_inv

    # Build cos(m*omega), sin(m*omega) via Chebyshev recurrence
    max_m = 5
    cos_m_tab = {0: hy.expression(1.0), 1: cos_omega}
    sin_m_tab = {0: hy.expression(0.0), 1: sin_omega}
    for m in range(2, max_m + 1):
        cos_m_tab[m] = 2.0 * cos_omega * cos_m_tab[m - 1] - cos_m_tab[m - 2]
        sin_m_tab[m] = 2.0 * cos_omega * sin_m_tab[m - 1] - sin_m_tab[m - 2]

    # --- Convert MEAN_DATA coefficients and build rate expressions ---
    _sympify_locals = {"q": sp.Symbol("q", real=True), "Q": sp.Symbol("Q", real=True), "I": sp.I}
    hy_var_map = {"q": q_ecl, "Q": Q_hy}

    rates = {}
    n_terms = 0
    for variable in ("g", "Q_rate", "Psi", "Omega", "M"):
        # Q_rate to avoid collision with Q_hy variable name
        var_key = "Q" if variable == "Q_rate" else variable
        rate_expr = hy.expression(0.0)

        for n_key in sorted(mean_data.keys(), key=int):
            n = int(n_key)
            coeffs = mean_data[n_key].get(var_key, {})

            for m_str, expr_str in coeffs.items():
                m = int(m_str)
                c_sp = sp.sympify(expr_str, locals=_sympify_locals)
                if c_sp == 0:
                    continue

                c_re_sp = sp.re(c_sp)
                c_im_sp = sp.im(c_sp)

                abs_m = abs(m)
                cos_mw = cos_m_tab[abs_m]
                sin_mw = sin_m_tab[abs_m]
                if m < 0 and abs_m > 0:
                    sin_mw = -sin_mw  # sin(-x) = -sin(x)

                # Re(c * exp(i*m*omega)) = Re(c)*cos(m*omega) - Im(c)*sin(m*omega)
                if c_re_sp != 0:
                    c_re_hy = _sympy_to_heyoka(c_re_sp, hy_var_map)
                    rate_expr = rate_expr + s_hy[n] * c_re_hy * cos_mw
                if c_im_sp != 0:
                    c_im_hy = _sympy_to_heyoka(c_im_sp, hy_var_map)
                    rate_expr = rate_expr - s_hy[n] * c_im_hy * sin_mw

                n_terms += 1

        rates[variable] = rate_expr

    # Scale by nu: physical rates = nu * dimensionless rates
    g_dot = nu_hy * rates["g"]
    Q_dot = nu_hy * rates["Q_rate"]
    Psi_dot = nu_hy * rates["Psi"]
    Omega_dot = nu_hy * rates["Omega"]
    M_dot = nu_hy * rates["M"]

    # --- Convert to (p1, p2, q1, q2) derivatives ---
    # p1_dot = g_dot * p1/g + p2 * Psi_dot
    # p2_dot = g_dot * p2/g - p1 * Psi_dot
    # q1_dot = Q_dot * q1/Q + q2 * Omega_dot
    # q2_dot = Q_dot * q2/Q - q1 * Omega_dot
    g_inv = 1.0 / g_hy
    Q_inv = 1.0 / Q_hy

    p1_dot_expr = g_dot * p1 * g_inv + p2 * Psi_dot
    p2_dot_expr = g_dot * p2 * g_inv - p1 * Psi_dot
    M_dot_total = nu_hy + M_dot
    q1_dot_expr = Q_dot * q1 * Q_inv + q2 * Omega_dot
    q2_dot_expr = Q_dot * q2 * Q_inv - q1 * Omega_dot

    t_build = time.time() - t_start

    # --- Compile ---
    t_compile_start = time.time()
    cf = hy.cfunc(
        [p1_dot_expr, p2_dot_expr, M_dot_total, q1_dot_expr, q2_dot_expr],
        vars=[p1, p2, q1, q2],
        compact_mode=True,
    )
    t_compile = time.time() - t_compile_start

    info = {
        "n_terms": n_terms,
        "build_time": t_build,
        "compile_time": t_compile,
    }
    return cf, info


def get_mean_rhs_cfunc():
    """Lazy-build and cache the mean RHS cfunc."""
    global _CACHED_CFUNC, _CACHED_CFUNC_INFO
    if _CACHED_CFUNC is not None:
        return _CACHED_CFUNC, _CACHED_CFUNC_INFO
    cf, info = build_mean_rhs_cfunc()
    _CACHED_CFUNC = cf
    _CACHED_CFUNC_INFO = info
    return cf, info


# ---------------------------------------------------------------------------
#  RK4 integrator using the compiled cfunc
# ---------------------------------------------------------------------------

def rk4_integrate_mean_compiled(
    state0: np.ndarray,
    t_eval: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float,
    mu_val: float,
    substeps: int = 8,
) -> np.ndarray:
    """RK4 integration of mean GEqOE slow flow using heyoka cfunc.

    Parameters
    ----------
    state0 : [nu, p1, p2, M, q1, q2] mean state
    t_eval : time grid [s]
    j_coeffs : {degree: Jn_value}
    substeps : RK4 substeps per interval

    Returns
    -------
    states : (N, 6) array of mean states at each time
    """
    cf, _ = get_mean_rhs_cfunc()

    nu_val = float(state0[0])
    a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)
    ra = re_val / a_val
    ra2 = ra * ra

    # Parameters: [nu, s2, s3, s4, s5]
    pars = np.array([
        nu_val,
        j_coeffs[2] * ra2,
        j_coeffs[3] * ra2 * ra,
        j_coeffs[4] * ra2 * ra2,
        j_coeffs[5] * ra2 * ra2 * ra,
    ])

    N = len(t_eval)
    out = np.empty((N, 6))
    out[0] = state0

    # Working arrays for cfunc: input [p1, p2, q1, q2], output [5 rates]
    inp = np.empty(4)
    rhs_out = np.empty(5)

    # State: [p1, p2, M, q1, q2] — 5 active DOFs (nu constant)
    y = np.array([state0[1], state0[2], state0[3], state0[4], state0[5]])

    def rhs(y_vec):
        inp[0] = y_vec[0]  # p1
        inp[1] = y_vec[1]  # p2
        inp[2] = y_vec[3]  # q1
        inp[3] = y_vec[4]  # q2
        cf(inp, pars=pars, outputs=rhs_out)
        # rhs_out = [p1_dot, p2_dot, M_dot_total, q1_dot, q2_dot]
        return rhs_out  # [p1_dot, p2_dot, M_dot, q1_dot, q2_dot]

    k1 = np.empty(5)
    k2 = np.empty(5)
    k3 = np.empty(5)
    k4 = np.empty(5)
    y_tmp = np.empty(5)

    for i in range(N - 1):
        dt = (t_eval[i + 1] - t_eval[i]) / substeps
        for _ in range(substeps):
            k1[:] = rhs(y)

            y_tmp[:] = y + 0.5 * dt * k1
            k2[:] = rhs(y_tmp)

            y_tmp[:] = y + 0.5 * dt * k2
            k3[:] = rhs(y_tmp)

            y_tmp[:] = y + dt * k3
            k4[:] = rhs(y_tmp)

            y[:] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        out[i + 1, 0] = nu_val
        out[i + 1, 1:] = y

    return out


# ---------------------------------------------------------------------------
#  Adaptive integrator (scipy DOP853) using the compiled cfunc
# ---------------------------------------------------------------------------

def adaptive_integrate_mean_compiled(
    state0: np.ndarray,
    t_eval: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float,
    mu_val: float,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Adaptive DOP853 integration of mean GEqOE slow flow using heyoka cfunc.

    Uses scipy's 8th-order Dormand-Prince adaptive integrator with the
    LLVM-compiled cfunc as the RHS.  For smooth averaged dynamics this
    typically needs far fewer steps than fixed-step RK4.

    Parameters
    ----------
    state0 : [nu, p1, p2, M, q1, q2] mean state
    t_eval : time grid [s]
    j_coeffs : {degree: Jn_value}
    rtol, atol : integrator tolerances

    Returns
    -------
    states : (N, 6) array of mean states at each time
    """
    from scipy.integrate import solve_ivp

    cf, _ = get_mean_rhs_cfunc()

    nu_val = float(state0[0])
    a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)
    ra = re_val / a_val
    ra2 = ra * ra

    pars = np.array([
        nu_val,
        j_coeffs[2] * ra2,
        j_coeffs[3] * ra2 * ra,
        j_coeffs[4] * ra2 * ra2,
        j_coeffs[5] * ra2 * ra2 * ra,
    ])

    # Working arrays (pre-allocated, reused per call)
    inp = np.empty(4)
    rhs_out = np.empty(5)

    def rhs_scipy(_t, y_vec):
        inp[0] = y_vec[0]  # p1
        inp[1] = y_vec[1]  # p2
        inp[2] = y_vec[3]  # q1
        inp[3] = y_vec[4]  # q2
        cf(inp, pars=pars, outputs=rhs_out)
        return rhs_out.copy()  # scipy needs a fresh array each call

    y0 = np.array([state0[1], state0[2], state0[3], state0[4], state0[5]])

    sol = solve_ivp(
        rhs_scipy,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    N = len(t_eval)
    out = np.empty((N, 6))
    out[:, 0] = nu_val
    out[:, 1:] = sol.y.T  # sol.y is (5, N), transpose to (N, 5)

    return out
