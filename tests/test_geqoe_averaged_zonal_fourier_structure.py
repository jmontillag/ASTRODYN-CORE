from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from astrodyn_core.geqoe_taylor import J2, J3, J4, J5

# Add geqoe_mean package to path
_DOC_DIR = Path(__file__).resolve().parents[1] / "docs" / "geqoe_averaged"
if str(_DOC_DIR) not in sys.path:
    sys.path.insert(0, str(_DOC_DIR))


def test_total_order_zonal_fit_closes_in_expected_basis() -> None:
    from geqoe_mean.fourier_model import project_total_order_model

    omega_grid = np.linspace(0.0, 2.0 * np.pi, 49, dtype=float)[:-1]
    fitted = project_total_order_model(
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
