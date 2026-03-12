from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")

from astrodyn_core.geqoe_taylor import J3, J4, MU, RE, ZonalPerturbation

# Add geqoe_mean package and scripts to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_DIR = _REPO_ROOT / "docs" / "geqoe_averaged"
_SCRIPT_DIR = _DOC_DIR / "scripts"
if str(_DOC_DIR) not in sys.path:
    sys.path.insert(0, str(_DOC_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _load_script(rel_path: str, module_name: str):
    module_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_symbolic_j3_j4_coefficients_match_exact_averaged_drift() -> None:
    from geqoe_mean.fourier_model import avg_slow_drift, frozen_state
    coeff_mod = _load_script("docs/geqoe_averaged/scripts/zonal_symbolic_coeffs.py", "zonal_symbolic_coeffs")

    f_G31 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.G31, "numpy")
    f_Q31 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.Q31, "numpy")
    f_P31 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.P31, "numpy")
    f_O31 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.O31, "numpy")
    f_G42 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.G42, "numpy")
    f_Q42 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.Q42, "numpy")
    f_P40 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.P40, "numpy")
    f_P42 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.P42, "numpy")
    f_O40 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.O40, "numpy")
    f_O42 = sympy.lambdify((coeff_mod.q, coeff_mod.Q), coeff_mod.O42, "numpy")

    a_km = 16000.0
    e = 0.35
    inc_deg = 50.0
    raan_deg = 25.0
    beta = np.sqrt(1.0 - e * e)
    q_val = e / (1.0 + beta)
    Q_val = np.tan(np.deg2rad(inc_deg) / 2.0)
    nu = np.sqrt(MU / a_km**3)
    eps3 = nu * J3 * (RE / a_km) ** 3
    eps4 = nu * J4 * (RE / a_km) ** 4

    omega_grid = np.deg2rad([20.0, 45.0, 100.0, 160.0])

    for omega in omega_grid:
        state = frozen_state(a_km, e, inc_deg, raan_deg, np.rad2deg(omega))

        drift_j3 = avg_slow_drift(state, ZonalPerturbation({3: J3}))
        assert drift_j3["g_dot"] == pytest.approx(eps3 * f_G31(q_val, Q_val) * np.cos(omega), rel=1e-10, abs=1e-14)
        assert drift_j3["Q_dot"] == pytest.approx(eps3 * f_Q31(q_val, Q_val) * np.cos(omega), rel=1e-10, abs=1e-14)
        assert drift_j3["Psi_dot"] == pytest.approx(eps3 * f_P31(q_val, Q_val) * np.sin(omega), rel=1e-10, abs=1e-14)
        assert drift_j3["Omega_dot"] == pytest.approx(eps3 * f_O31(q_val, Q_val) * np.sin(omega), rel=1e-10, abs=1e-14)

        drift_j4 = avg_slow_drift(state, ZonalPerturbation({4: J4}))
        assert drift_j4["g_dot"] == pytest.approx(eps4 * f_G42(q_val, Q_val) * np.sin(2.0 * omega), rel=1e-10, abs=1e-14)
        assert drift_j4["Q_dot"] == pytest.approx(eps4 * f_Q42(q_val, Q_val) * np.sin(2.0 * omega), rel=1e-10, abs=1e-14)
        assert drift_j4["Psi_dot"] == pytest.approx(
            eps4 * (f_P40(q_val, Q_val) + f_P42(q_val, Q_val) * np.cos(2.0 * omega)),
            rel=1e-10,
            abs=1e-14,
        )
        assert drift_j4["Omega_dot"] == pytest.approx(
            eps4 * (f_O40(q_val, Q_val) + f_O42(q_val, Q_val) * np.cos(2.0 * omega)),
            rel=1e-10,
            abs=1e-14,
        )
