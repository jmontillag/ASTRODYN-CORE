# Averaged GEqOE Scripts

## Dependency graph

```
zonal_symbolic_general.py
  └── zonal_symbolic_coeffs.py         (imports symbolic machinery)
      └── zonal_short_period_general.py (imports symbolic machinery)
          └── zonal_short_period_generated.py  (generated data file)
              ├── zonal_short_period_validation.py
              ├── zonal_mean_validation.py
              ├── run_full_validation.py
              ├── extended_validation.py
              └── analytical_pipeline_figure.py
short_period_direct.py                 (alternative solver, imports zonal_short_period_general)
zonal_fourier_model.py                 (standalone Fourier approach)
zonal_harmonic_probe.py                (standalone analysis)
j2_secular_vs_osculating.py            (standalone J2 comparison)
validation_check.py                    (standalone J2 validation)
```

## Run order for full regeneration

1. **Symbolic generation** (slow — J5 takes ~80 min):
   ```bash
   python scripts/zonal_symbolic_general.py
   ```
2. **Coefficient LaTeX** (writes zonal_symbolic_coeffs.tex):
   ```bash
   python scripts/zonal_symbolic_coeffs.py
   ```
3. **Short-period coefficients** (generates zonal_short_period_generated.py):
   ```bash
   python scripts/zonal_short_period_general.py
   ```
4. **Validation** (any order after step 3):
   ```bash
   python scripts/zonal_mean_validation.py
   python scripts/zonal_short_period_validation.py
   python scripts/run_full_validation.py
   python scripts/extended_validation.py
   ```
5. **Figures** (any order after step 3):
   ```bash
   python scripts/analytical_pipeline_figure.py
   python scripts/j2_secular_vs_osculating.py
   ```

All commands should be prefixed with `conda run -n astrodyn-core-env`.

## Per-script details

| Script | Purpose | Key inputs | Key outputs | Runtime |
|--------|---------|-----------|-------------|---------|
| `zonal_symbolic_general.py` | Exact degree-n symbolic averaged GEqOE via Laurent/Legendre expansion | Degree n, GEqOE symbols | Symbolic mean rates, SP coefficients | ~1 min (J3), ~80 min (J5) |
| `zonal_symbolic_coeffs.py` | Write explicit J3–J5 coefficient formulas to LaTeX | Symbolic machinery from above | `zonal_symbolic_coeffs.tex` | ~5 min |
| `zonal_short_period_general.py` | Generate exact first-order mixed-zonal short-period map | Symbolic averaged rates | `zonal_short_period_generated.py` | ~20 min |
| `zonal_short_period_generated.py` | Generated coefficient data (not runnable) | — | Imported by validation scripts | — |
| `zonal_fourier_model.py` | Finite Fourier averaged zonal model construction + validation | GEqOE ICs | Console output, diagnostics | ~2 min |
| `zonal_harmonic_probe.py` | Probe harmonic structure of averaged zonal drift | GEqOE ICs | Console analysis | ~1 min |
| `zonal_mean_validation.py` | Validate exact zonal mean model (pointwise + long-arc) | GEqOE ICs, zonal model | Figures, console | ~3 min |
| `zonal_short_period_validation.py` | Validate short-period map (Cartesian reconstruction errors) | Generated coefficients | `figures/zonal_short_period_*.png` | ~5 min |
| `short_period_direct.py` | Direct residue-based short-period solver (alternative method) | Symbolic expressions | Console comparison | ~10 min |
| `j2_secular_vs_osculating.py` | Compare closed secular J2 vs full osculating integration | GEqOE ICs | `figures/j2_secular_*.png` | ~1 min |
| `validation_check.py` | Comprehensive averaged J2 validation | GEqOE ICs | Console output | ~2 min |
| `run_full_validation.py` | Full J2–J4/J5 validation suite | Generated coefficients | Console + figures | ~5 min |
| `extended_validation.py` | Extended validation across 12 orbital regimes | Generated coefficients | `figures/extended_*.png`, JSON | ~10 min |
| `analytical_pipeline_figure.py` | Orbital element decomposition figure | CSV data, GEqOE ICs | `figures/analytical_pipeline.png` | ~2 min |

## Environment

- **Conda environment**: `astrodyn-core-env`
- **Key packages**: heyoka ≥ 7.9.2, sympy, numpy, matplotlib
- **Optional**: orekit (for Brouwer comparison in `extended_validation.py`)
