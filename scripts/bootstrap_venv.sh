#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
if [ "${BOOTSTRAP_UPGRADE_BUILD_TOOLS:-0}" = "1" ]; then
  python -m pip install --default-timeout "${PIP_TIMEOUT:-1000}" --retries "${PIP_RETRIES:-10}" --upgrade pip wheel setuptools
fi
python -m pip install --default-timeout "${PIP_TIMEOUT:-1000}" --retries "${PIP_RETRIES:-10}" -r "${ROOT_DIR}/requirements.txt"
