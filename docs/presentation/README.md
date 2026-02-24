# ASTRODYN-CORE Beamer Presentation

This folder contains a LaTeX Beamer presentation showcasing the capabilities of ASTRODYN-CORE, a next-generation orbital propagation framework.

## Contents

- `astrodyn_presentation.tex` - Main Beamer presentation source
- `plots/` - Generated plots for the presentation
  - `generate_plots.py` - Script to generate fidelity and orbital elements plots
  - `generate_geqoe_benchmark.py` - Script to generate GEqOE benchmark plot
  - `generate_covariance.py` - Script to generate covariance growth plot

## Generating the Presentation

### 1. Generate Plots

First, generate the plots used in the presentation:

```bash
# Generate fidelity comparison and orbital elements plots
cd docs/presentation/plots
python generate_plots.py

# Generate GEqOE benchmark plot
python generate_geqoe_benchmark.py

# Generate covariance growth plot
python generate_covariance.py
```

Or generate all plots at once:

```bash
cd docs/presentation/plots
python generate_plots.py && python generate_geqoe_benchmark.py && python generate_covariance.py
```

Note: These scripts require the `astrodyn-core-env` conda environment to be active.

### 2. Build PDF

Compile the LaTeX presentation:

```bash
cd docs/presentation
pdflatex -interaction=nonstopmode astrodyn_presentation.tex
```

For best results, run `pdflatex` twice to resolve references.

## Presentation Structure

1. **Introduction** - What is ASTRODYN-CORE
2. **Quick Start** - Ease of use examples with code
3. **Propagation Examples** - Keplerian, Numerical, DSST, TLE
4. **Custom Propagator: GEqOE** - J2 Taylor-series propagator
5. **Derivatives and Covariance** - STM-based uncertainty propagation
6. **State Files and I/O** - YAML/JSON/HDF5 workflows
7. **Future Features** - Derivatives wrt parameters, Field-based propagators
8. **Conclusion** - Summary and getting started

## Dependencies

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Python environment with:
  - matplotlib
  - numpy
  - astrodyn_core (with Orekit)

## Running Examples

To see the examples in action:

```bash
# Quick start
python examples/quickstart.py --mode all

# GEqOE propagator
python examples/geqoe_propagator.py --mode all

# Covariance/uncertainty
python examples/uncertainty.py

# Multi-fidelity comparison
python examples/cookbook/multi_fidelity_comparison.py
```
