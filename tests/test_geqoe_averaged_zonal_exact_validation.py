from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_exact_truncated_zonal_rhs_matches_numeric_average() -> None:
    mod = _load_module("docs/geqoe_averaged/zonal_mean_validation.py", "zonal_mean_validation")
    case = mod.ValidationCase(n_orbits=6, samples_per_orbit=48, rk4_substeps_per_orbit=8)
    j_coeffs = {2: mod.J2, 3: mod.J3, 4: mod.J4, 5: mod.J5}
    pointwise = mod.pointwise_rhs_validation(case, j_coeffs, omega_samples=12)
    scaled = mod.pointwise_rhs_validation(case, {n: 0.1 * val for n, val in j_coeffs.items()}, omega_samples=12)

    assert pointwise["g_dot"] < 1.0e-10
    assert pointwise["Q_dot"] < 5.0e-3
    assert pointwise["Psi_dot"] < 5.0e-4
    assert pointwise["Omega_dot"] < 1.0e-4

    assert scaled["Q_dot"] < 0.2 * pointwise["Q_dot"]
    assert scaled["Psi_dot"] < 0.2 * pointwise["Psi_dot"]
    assert scaled["Omega_dot"] < 0.2 * pointwise["Omega_dot"]


def test_exact_truncated_zonal_mean_flow_tracks_orbit_means() -> None:
    mod = _load_module("docs/geqoe_averaged/zonal_mean_validation.py", "zonal_mean_validation_flow")
    case = mod.ValidationCase(n_orbits=10, samples_per_orbit=48, rk4_substeps_per_orbit=8)
    j_coeffs = {2: mod.J2, 3: mod.J3, 4: mod.J4, 5: mod.J5}
    result = mod.propagate_full_and_mean(case, j_coeffs)
    metrics = result["metrics"]

    assert metrics["p1_rel_rms"] < 5.0e-4
    assert metrics["p2_rel_rms"] < 5.0e-4
    assert metrics["q1_rel_rms"] < 5.0e-4
    assert metrics["q2_rel_rms"] < 5.0e-4
    assert metrics["psi_rms"] < 1.0e-3
    assert metrics["Omega_rms"] < 1.0e-3
