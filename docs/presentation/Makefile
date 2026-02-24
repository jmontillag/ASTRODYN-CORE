# Makefile for ASTRODYN-CORE Beamer Presentation
SHELL := /bin/bash

.PHONY: all plots pdf clean help

PROJECT_ROOT := $(shell git -C . rev-parse --show-toplevel 2>/dev/null || echo ../..)

# Activate conda before every python / pdflatex call
CONDA_RUN = source ~/miniconda3/etc/profile.d/conda.sh && conda activate astrodyn-core-env &&

# Default target
all: plots pdf

# Help target
help:
	@echo "ASTRODYN-CORE Presentation Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all     - Generate plots and build PDF (default)"
	@echo "  plots   - Generate all plots"
	@echo "  pdf     - Build the PDF presentation"
	@echo "  clean   - Remove generated files"
	@echo "  help    - Show this help message"

# Generate all plots (run from project root so orekit-data.zip is found)
plots:
	@echo "Generating plots..."
	@cd $(PROJECT_ROOT) && $(CONDA_RUN) python docs/presentation/plots/generate_plots.py
	@cd $(PROJECT_ROOT) && $(CONDA_RUN) python docs/presentation/plots/generate_geqoe_benchmark.py
	@cd $(PROJECT_ROOT) && $(CONDA_RUN) python docs/presentation/plots/generate_covariance.py
	@echo "Plots generated successfully!"

# Build PDF (needs conda so pygmentize is on PATH for minted)
pdf: astrodyn_presentation.pdf

astrodyn_presentation.pdf: astrodyn_presentation.tex
	@echo "Building PDF..."
	$(CONDA_RUN) pdflatex -shell-escape -interaction=nonstopmode astrodyn_presentation.tex || true
	$(CONDA_RUN) pdflatex -shell-escape -interaction=nonstopmode astrodyn_presentation.tex
	@echo "PDF built successfully!"

# Clean generated files
clean:
	@echo "Cleaning..."
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.w18 *.pyg
	@rm -rf _minted-astrodyn_presentation
	@rm -f astrodyn_presentation.pdf
	@rm -f plots/*.png
	@echo "Clean complete!"
