# Averaged GEqOE Scripts

All scripts import from the `geqoe_mean` package (`../geqoe_mean/`). They are
thin entry points for symbolic generation, validation, and figure creation.

## Dependency graph

```
geqoe_mean/                              # importable package
  ├── symbolic.py                        # degree-n symbolic engine
  ├── short_period.py                    # short-period map
  ├── generated_coefficients.py          # pre-computed J2-J5 data
  ├── fourier_model.py                   # frozen-state numerical averaging
  ├── direct_residue.py                  # direct residue solver
  ├── coordinates.py                     # kepler_to_rv, rotations
  ├── constants.py                       # J_COEFFS, MU, RE
  └── validation.py                      # RK4, error metrics

scripts/
  ├── zonal_symbolic_coeffs.py           # → geqoe_mean.symbolic
  ├── zonal_mean_validation.py           # → geqoe_mean.symbolic, fourier_model
  ├── zonal_short_period_validation.py   # → geqoe_mean.short_period, validation
  ├── run_full_validation.py             # → zonal_short_period_validation
  ├── extended_validation.py             # → geqoe_mean.short_period, validation
  ├── scaling_diagnostic.py             # → zonal_short_period_validation, geqoe_mean
  ├── analytical_pipeline_figure.py      # → geqoe_mean.short_period, validation
  ├── j2_secular_vs_osculating.py        # → geqoe_mean.coordinates
  └── validation_check.py               # → geqoe_mean.coordinates
```

## Run order for full regeneration

1. **Short-period coefficient generation** (slow — J5 takes ~80 min):
   ```bash
   python geqoe_mean/direct_residue.py --generate
   ```
   This regenerates `geqoe_mean/generated_coefficients.py`.

2. **Coefficient LaTeX** (writes zonal_symbolic_coeffs.tex):
   ```bash
   python scripts/zonal_symbolic_coeffs.py
   ```

3. **Validation** (any order after step 1):
   ```bash
   python scripts/zonal_mean_validation.py
   python scripts/zonal_short_period_validation.py
   python scripts/run_full_validation.py
   python scripts/extended_validation.py
   ```

4. **Figures** (any order after step 1):
   ```bash
   python scripts/analytical_pipeline_figure.py
   python scripts/j2_secular_vs_osculating.py
   ```

All commands should be prefixed with `conda run -n astrodyn-core-env` and
run from the repository root.

## Per-script details

| Script | Purpose | Key outputs | Runtime |
|--------|---------|-------------|---------|
| `zonal_symbolic_coeffs.py` | Write explicit J3–J5 coefficient formulas to LaTeX | `zonal_symbolic_coeffs.tex` | ~5 min |
| `zonal_mean_validation.py` | Validate exact zonal mean model (pointwise + long-arc) | Figures, console | ~3 min |
| `zonal_short_period_validation.py` | Validate short-period map (Cartesian reconstruction) | `figures/zonal_short_period_*.png` | ~5 min |
| `j2_secular_vs_osculating.py` | Compare closed secular J2 vs full osculating | `figures/j2_secular_*.png` | ~1 min |
| `validation_check.py` | Comprehensive averaged J2 validation | Console output | ~2 min |
| `run_full_validation.py` | Full J2–J4/J5 validation suite | Console + figures | ~5 min |
| `extended_validation.py` | Extended validation across 12 orbital regimes | `figures/extended_*.png`, JSON | ~10 min |
| `scaling_diagnostic.py` | Diagnose λ-scaling slope (pointwise vs arc-integrated) | `figures/scaling_diagnostic.png` | ~3 min |
| `analytical_pipeline_figure.py` | Orbital element decomposition figure | `figures/analytical_pipeline.png` | ~2 min |

## Environment

- **Conda environment**: `astrodyn-core-env`
- **Key packages**: heyoka >= 7.9.2, sympy, numpy, matplotlib
- **Optional**: orekit (for Brouwer comparison in `extended_validation.py`)
