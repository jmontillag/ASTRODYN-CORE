from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")

from astrodyn_core.geqoe_taylor import J3, J4, MU, RE, ZonalPerturbation


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evaluate_harmonic_model(
    coeffs: dict[str, dict[int, object]],
    q_val: float,
    Q_val: float,
    omega: float,
    eps_n: float,
) -> dict[str, float]:
    out = {}
    parity_odd = any(k % 2 for k in coeffs["g"])
    lambda_cache: dict[tuple[str, int], object] = {}

    def eval_terms(term_map: dict[int, object], trig: str, include_constant: bool = False) -> float:
        total = 0.0
        for m, expr in term_map.items():
            key = (sympy.srepr(expr), m)
            f = lambda_cache.get(key)
            if f is None:
                f = sympy.lambdify(("q", "Q"), expr, "numpy")
                lambda_cache[key] = f
            if m == 0 and include_constant:
                total += float(f(q_val, Q_val))
                continue
            if trig == "cos":
                total += float(f(q_val, Q_val)) * np.cos(m * omega)
            else:
                total += float(f(q_val, Q_val)) * np.sin(m * omega)
        return eps_n * total

    if parity_odd:
        out["g_dot"] = eval_terms(coeffs["g"], "cos")
        out["Q_dot"] = eval_terms(coeffs["Q"], "cos")
        out["Psi_dot"] = eval_terms(coeffs["Psi"], "sin")
        out["Omega_dot"] = eval_terms(coeffs["Omega"], "sin")
    else:
        out["g_dot"] = eval_terms(coeffs["g"], "sin")
        out["Q_dot"] = eval_terms(coeffs["Q"], "sin")
        out["Psi_dot"] = eval_terms(coeffs["Psi"], "cos", include_constant=True)
        out["Omega_dot"] = eval_terms(coeffs["Omega"], "cos", include_constant=True)
    return out


def test_general_symbolic_generator_matches_exact_averaged_drift() -> None:
    fourier_mod = _load_module("docs/geqoe_averaged/scripts/zonal_fourier_model.py", "zonal_fourier_model")
    general_mod = _load_module("docs/geqoe_averaged/scripts/zonal_symbolic_general.py", "zonal_symbolic_general")

    a_km = 16000.0
    e = 0.35
    inc_deg = 50.0
    raan_deg = 25.0
    beta = np.sqrt(1.0 - e * e)
    q_val = e / (1.0 + beta)
    Q_val = np.tan(np.deg2rad(inc_deg) / 2.0)
    nu = np.sqrt(MU / a_km**3)

    cases = {
        3: J3,
        4: J4,
    }
    omega_grid = np.deg2rad([20.0, 45.0, 100.0, 160.0])

    for degree, Jn in cases.items():
        coeffs = general_mod.harmonic_coefficients(degree)
        eps_n = nu * Jn * (RE / a_km) ** degree
        for omega in omega_grid:
            state = fourier_mod.frozen_state(a_km, e, inc_deg, raan_deg, np.rad2deg(omega))
            drift = fourier_mod.avg_slow_drift(state, ZonalPerturbation({degree: Jn}))
            expected = _evaluate_harmonic_model(coeffs, q_val, Q_val, omega, eps_n)

            assert drift["g_dot"] == pytest.approx(expected["g_dot"], rel=1e-10, abs=1e-14)
            assert drift["Q_dot"] == pytest.approx(expected["Q_dot"], rel=1e-10, abs=1e-14)
            assert drift["Psi_dot"] == pytest.approx(expected["Psi_dot"], rel=1e-10, abs=1e-14)
            assert drift["Omega_dot"] == pytest.approx(expected["Omega_dot"], rel=1e-10, abs=1e-14)


def test_isolated_degree_max_harmonic_is_n_minus_2() -> None:
    general_mod = _load_module("docs/geqoe_averaged/scripts/zonal_symbolic_general.py", "zonal_symbolic_general_harmonics")

    for degree in (3, 4):
        coeffs = general_mod.harmonic_coefficients(degree)
        g_keys = sorted(k for k in coeffs["g"] if k != 0)
        q_keys = sorted(k for k in coeffs["Q"] if k != 0)
        psi_keys = sorted(k for k in coeffs["Psi"] if k != 0)
        omega_keys = sorted(k for k in coeffs["Omega"] if k != 0)

        if degree % 2:
            assert max(g_keys) == degree - 2
            assert max(q_keys) == degree - 2
            assert max(psi_keys) == degree - 2
            assert max(omega_keys) == degree - 2
        else:
            assert max(g_keys) == degree - 2
            assert max(q_keys) == degree - 2
            assert max(psi_keys) == degree - 2
            assert max(omega_keys) == degree - 2
