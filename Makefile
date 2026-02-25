SHELL := /bin/bash

CONDA_ENV ?= astrodyn-core-env
CONDA_RUN := conda run -n $(CONDA_ENV)
BUILD_DIR ?= build

.PHONY: help setup-env install-dev test test-rs test-fast test-transition \
	docs-install docs-serve docs-build \
	example-quickstart example-geqoe native-config native-build

help:
	@echo "ASTRODYN-CORE developer shortcuts"
	@echo ""
	@echo "Environment:"
	@echo "  make setup-env             # run repo setup helper"
	@echo "  make install-dev           # pip install -e .[dev] in $(CONDA_ENV)"
	@echo "  make docs-install          # install docs tooling in $(CONDA_ENV)"
	@echo ""
	@echo "Tests:"
	@echo "  make test                 # full pytest -q"
	@echo "  make test-rs              # pytest -q -rs (show skips)"
	@echo "  make test-fast            # quick local gate"
	@echo "  make test-transition      # architecture hardening transition gate"
	@echo ""
	@echo "Examples:"
	@echo "  make example-quickstart   # run examples/quickstart.py --mode all"
	@echo "  make example-geqoe        # run examples/geqoe_propagator.py --mode all"
	@echo ""
	@echo "Docs:"
	@echo "  make docs-serve           # serve MkDocs site locally"
	@echo "  make docs-build           # build MkDocs site"
	@echo ""
	@echo "Native build:"
	@echo "  make native-config        # cmake -S . -B $(BUILD_DIR)"
	@echo "  make native-build         # cmake --build $(BUILD_DIR)"

setup-env:
	python setup_env.py

install-dev:
	$(CONDA_RUN) python -m pip install -e .[dev]

docs-install:
	$(CONDA_RUN) python -m pip install -e .[docs]

test:
	$(CONDA_RUN) pytest -q

test-rs:
	$(CONDA_RUN) pytest -q -rs

test-fast:
	$(CONDA_RUN) pytest -q tests/test_api_boundary_hygiene.py tests/test_registry_factory.py

test-transition:
	$(CONDA_RUN) pytest -q -rs \
		tests/test_universe_config.py \
		tests/test_dsst_assembly.py \
		tests/test_registry_factory.py \
		tests/test_api_boundary_hygiene.py

example-quickstart:
	$(CONDA_RUN) python examples/quickstart.py --mode all

example-geqoe:
	$(CONDA_RUN) python examples/geqoe_propagator.py --mode all

docs-serve:
	$(CONDA_RUN) mkdocs serve

docs-build:
	$(CONDA_RUN) mkdocs build --strict

native-config:
	cmake -S . -B $(BUILD_DIR)

native-build:
	cmake --build $(BUILD_DIR)
