from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from astrodyn_core.geqoe_taylor import J2


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_short_period_mean_rates_match_existing_symbolic_slow_model() -> None:
    short_mod = _load_module(
        "docs/geqoe_averaged/scripts/zonal_short_period_general.py",
        "zonal_short_period_general_test",
    )
    slow_mod = _load_module(
        "docs/geqoe_averaged/scripts/zonal_symbolic_general.py",
        "zonal_symbolic_general_test",
    )

    nu = np.sqrt(short_mod.MU / 16000.0**3)
    g_val = 0.35
    Q_val = np.tan(np.deg2rad(50.0) / 2.0)
    omega = np.deg2rad(40.0)

    degree = 2
    jn = J2
    new = short_mod.evaluate_isolated_degree_mean_rates(
        degree,
        nu_val=nu,
        g_val=g_val,
        Q_val=Q_val,
        omega_val=omega,
        jn_val=jn,
        re_val=short_mod.RE,
        mu_val=short_mod.MU,
    )
    old = slow_mod.evaluate_isolated_degree_mean_rates(
        degree,
        nu_val=nu,
        g_val=g_val,
        Q_val=Q_val,
        omega_val=omega,
        jn_val=jn,
        re_val=short_mod.RE,
        mu_val=short_mod.MU,
    )

    assert new["g_dot"] == pytest.approx(old["g_dot"], rel=1.0e-10, abs=1.0e-14)
    assert new["Q_dot"] == pytest.approx(old["Q_dot"], rel=1.0e-10, abs=1.0e-14)
    assert new["Psi_dot"] == pytest.approx(old["Psi_dot"], rel=1.0e-10, abs=1.0e-14)
    assert new["Omega_dot"] == pytest.approx(old["Omega_dot"], rel=1.0e-10, abs=1.0e-14)
    assert abs(new["M_dot"]) < 1.0e-14


def test_j2_mean_anomaly_short_period_support_is_finite() -> None:
    mod = _load_module(
        "docs/geqoe_averaged/scripts/zonal_short_period_general.py",
        "zonal_short_period_general_support",
    )

    coeffs = mod.isolated_short_period_coefficients_for("M", 2)
    m_support = sorted({m_val for (m_val, _) in coeffs})
    k_support = sorted({k_val for (_, k_val) in coeffs})

    assert m_support == [-2, 0, 2]
    assert min(k_support) >= 0
    assert max(k_support) <= 3


def test_mean_rhs_retains_keplerian_mean_anomaly_advance() -> None:
    mod = _load_module(
        "docs/geqoe_averaged/scripts/zonal_short_period_general.py",
        "zonal_short_period_general_rhs",
    )

    state_mean = np.array([np.sqrt(mod.MU / 9000.0**3), 0.02, 0.04, 0.3, 0.1, 0.2], dtype=float)
    rhs = mod.evaluate_truncated_mean_rhs_pqm(state_mean, {2: J2})

    assert rhs[0] == 0.0
    assert rhs[3] > 0.0
    assert rhs[3] == pytest.approx(state_mean[0], rel=1.0e-3)


def test_empty_generated_short_period_entry_is_treated_as_missing() -> None:
    mod = _load_module(
        "docs/geqoe_averaged/scripts/zonal_short_period_general.py",
        "zonal_short_period_general_generated_empty",
    )

    mod._SHORT_DATA_STR = {2: {"Psi": {}}}
    mod._generated_short_period_expressions.cache_clear()

    assert mod._generated_short_period_expressions("Psi", 2) is None
