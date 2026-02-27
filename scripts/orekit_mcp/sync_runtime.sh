#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${OREKIT_MCP_SOURCE_DIR:-}" ]]; then
  SRC_DIR="${OREKIT_MCP_SOURCE_DIR}"
else
  DEFAULT_SOURCE_1="${HOME}/Projects/mcp-tools/tools/orekit_docs_mcp"
  DEFAULT_SOURCE_2="${REPO_ROOT}/tools/orekit_docs_mcp"
  if [[ -d "${DEFAULT_SOURCE_1}" ]]; then
    SRC_DIR="${DEFAULT_SOURCE_1}"
  else
    SRC_DIR="${DEFAULT_SOURCE_2}"
  fi
fi

RUNTIME_ROOT="${OREKIT_MCP_RUNTIME_ROOT:-${HOME}/.codex/mcp/orekit-docs}"
DST_DIR="${RUNTIME_ROOT}/tools/orekit_docs_mcp"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source directory not found: ${SRC_DIR}" >&2
  echo "Set OREKIT_MCP_SOURCE_DIR to the MCP source checkout path." >&2
  exit 1
fi

mkdir -p "${DST_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    "${SRC_DIR}/" "${DST_DIR}/"
else
  find "${DST_DIR}" -mindepth 1 -maxdepth 1 ! -name "__pycache__" -exec rm -rf {} +
  cp -a "${SRC_DIR}/." "${DST_DIR}/"
  find "${DST_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${DST_DIR}" -type f -name "*.pyc" -delete
fi

cat > "${RUNTIME_ROOT}/README.txt" <<EOF
# Orekit Docs MCP (Global Runtime Copy)

Runtime location: ${RUNTIME_ROOT}
Source sync origin: ${SRC_DIR}

This runtime copy is intended for Codex MCP server launch from any repository.
EOF

echo "Synced Orekit MCP runtime to: ${DST_DIR}"
echo "Runtime root: ${RUNTIME_ROOT}"
