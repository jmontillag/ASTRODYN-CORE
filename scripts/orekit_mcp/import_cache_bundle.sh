#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  import_cache_bundle.sh <bundle.tgz> [--replace] [--target <cache-root>]

Examples:
  import_cache_bundle.sh /path/to/orekit-cache-bundle.tgz
  import_cache_bundle.sh /path/to/orekit-cache-bundle.tgz --replace
  import_cache_bundle.sh /path/to/orekit-cache-bundle.tgz --target /home/user/.cache/orekit

Notes:
  - Default target is ~/.cache/orekit
  - If the target exists, the script refuses to overwrite unless --replace is set
  - With --replace, the existing target is moved to a timestamped backup
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

BUNDLE_PATH=""
TARGET_ROOT="${OREKIT_CACHE_ROOT:-${HOME}/.cache/orekit}"
REPLACE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replace)
      REPLACE=1
      shift
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "--target requires a value" >&2; exit 1; }
      TARGET_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "${BUNDLE_PATH}" ]]; then
        echo "Only one bundle path is allowed" >&2
        usage >&2
        exit 1
      fi
      BUNDLE_PATH="$1"
      shift
      ;;
  esac
done

if [[ -z "${BUNDLE_PATH}" ]]; then
  echo "Missing bundle path" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "Bundle not found: ${BUNDLE_PATH}" >&2
  exit 1
fi

if [[ -f "${BUNDLE_PATH}.sha256" ]] && command -v sha256sum >/dev/null 2>&1; then
  echo "Verifying checksum..."
  (cd "$(dirname "${BUNDLE_PATH}")" && sha256sum -c "$(basename "${BUNDLE_PATH}").sha256")
fi

TARGET_ROOT="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${TARGET_ROOT}")"
TARGET_PARENT="$(dirname "${TARGET_ROOT}")"

mkdir -p "${TARGET_PARENT}"

if [[ -e "${TARGET_ROOT}" ]]; then
  if [[ "${REPLACE}" -ne 1 ]]; then
    echo "Target already exists: ${TARGET_ROOT}" >&2
    echo "Re-run with --replace to move it aside and import the bundle." >&2
    exit 1
  fi
  BACKUP_PATH="${TARGET_ROOT}.backup-$(date +%Y%m%d-%H%M%S)"
  echo "Moving existing target to backup:"
  echo "  ${TARGET_ROOT} -> ${BACKUP_PATH}"
  mv "${TARGET_ROOT}" "${BACKUP_PATH}"
fi

echo "Importing Orekit cache bundle..."
echo "  bundle: ${BUNDLE_PATH}"
echo "  target parent: ${TARGET_PARENT}"

tar -C "${TARGET_PARENT}" -xzf "${BUNDLE_PATH}"

if [[ ! -d "${TARGET_ROOT}" ]]; then
  echo "Import finished but expected target not found: ${TARGET_ROOT}" >&2
  exit 1
fi

echo "Done."
echo "  restored cache: ${TARGET_ROOT}"

