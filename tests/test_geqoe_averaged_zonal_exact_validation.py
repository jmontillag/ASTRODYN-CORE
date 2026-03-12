from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_DIR = _REPO_ROOT / "docs" / "geqoe_averaged"
if str(_DOC_DIR) not in sys.path:
    sys.path.insert(0, str(_DOC_DIR))


def _load_script(rel_path: str, module_name: str):
    module_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_exact_truncated_zonal_rhs_matches_numeric_average() -> None:
    mod = _load_script("docs/geqoe_averaged/scripts/zonal_mean_validation.py", "zonal_mean_validation")
    from geqoe_mean.constants import J2, J3, J4, J5
    case = mod.ValidationCase(n_orbits=6, samples_per_orbit=48, rk4_substeps_per_orbit=8)
    j_coeffs = {2: J2, 3: J3, 4: J4, 5: J5}
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
    mod = _load_script("docs/geqoe_averaged/scripts/zonal_mean_validation.py", "zonal_mean_validation_flow")
    from geqoe_mean.constants import J2, J3, J4, J5
    case = mod.ValidationCase(n_orbits=10, samples_per_orbit=48, rk4_substeps_per_orbit=8)
    j_coeffs = {2: J2, 3: J3, 4: J4, 5: J5}
    result = mod.propagate_full_and_mean(case, j_coeffs)
    metrics = result["metrics"]

    assert metrics["p1_rel_rms"] < 5.0e-4
    assert metrics["p2_rel_rms"] < 5.0e-4
    assert metrics["q1_rel_rms"] < 5.0e-4
    assert metrics["q2_rel_rms"] < 5.0e-4
    assert metrics["psi_rms"] < 1.0e-3
    assert metrics["Omega_rms"] < 1.0e-3
