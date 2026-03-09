from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from astrodyn_core.geqoe_taylor import J2, J3, J4, J5


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "docs" / "geqoe_averaged" / "zonal_fourier_model.py"
    spec = importlib.util.spec_from_file_location("zonal_fourier_model", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_total_order_zonal_fit_closes_in_expected_basis() -> None:
    mod = _load_probe_module()

    omega_grid = np.linspace(0.0, 2.0 * np.pi, 49, dtype=float)[:-1]
    fitted = mod.project_total_order_model(
        a_km=16000.0,
        e=0.35,
        inc_deg=50.0,
        raan_deg=25.0,
        j_coeffs={2: J2, 3: J3, 4: J4, 5: J5},
        omega_grid=omega_grid,
        samples_k=2049,
    )

    for name in ("g_dot", "Q_dot", "Psi_dot", "Omega_dot"):
        _, _, rel_rms = fitted[name]
        assert rel_rms < 1.0e-8, f"{name} fit residual too large: {rel_rms:.3e}"
