#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${OREKIT_MCP_ENV_NAME:-mcp-tools-env}"
SPEC_PATH="${OREKIT_MCP_ENV_SPEC:-${REPO_ROOT}/docs/ai-config/environments/mcp-tools-env.yml}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"

if [[ -z "${CONDA_BIN}" ]]; then
  echo "conda not found in PATH. Set CONDA_BIN=/path/to/conda and retry." >&2
  exit 1
fi

if [[ ! -f "${SPEC_PATH}" ]]; then
  echo "Env spec not found: ${SPEC_PATH}" >&2
  exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Updating existing Conda env: ${ENV_NAME}"
  "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${SPEC_PATH}" --prune
else
  echo "Creating Conda env: ${ENV_NAME}"
  "${CONDA_BIN}" env create -f "${SPEC_PATH}"
fi

echo "Validating MCP package import in ${ENV_NAME}"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import mcp; print(mcp.__file__)"

echo
echo "Conda env ready: ${ENV_NAME}"
echo "Interpreter path (expected): ${HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
