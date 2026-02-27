#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CACHE_ROOT="${OREKIT_CACHE_ROOT:-${HOME}/.cache/orekit}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DEFAULT_OUT="${REPO_ROOT}/artifacts/orekit-cache-bundle-${TIMESTAMP}.tgz"
OUT_PATH="${1:-${OREKIT_CACHE_BUNDLE_OUT:-${DEFAULT_OUT}}}"

if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "Orekit cache root not found: ${CACHE_ROOT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PATH}")"

PARENT_DIR="$(dirname "${CACHE_ROOT}")"
BASENAME="$(basename "${CACHE_ROOT}")"

echo "Exporting cache bundle..."
echo "  source: ${CACHE_ROOT}"
echo "  output: ${OUT_PATH}"

tar -C "${PARENT_DIR}" -czf "${OUT_PATH}" "${BASENAME}"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${OUT_PATH}" > "${OUT_PATH}.sha256"
  echo "  sha256: ${OUT_PATH}.sha256"
fi

if command -v du >/dev/null 2>&1; then
  echo "  size: $(du -h "${OUT_PATH}" | awk '{print $1}')"
fi

echo "Done."

