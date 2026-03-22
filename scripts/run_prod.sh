#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -x "${ROOT_DIR}/.venv/bin/uvicorn" ]; then
  "${ROOT_DIR}/scripts/bootstrap_venv.sh"
fi

export ROST_AI_MODEL_DIR="${ROST_AI_MODEL_DIR:-${ROOT_DIR}/models/best_openvino_model}"
export ROST_AI_ATTACHMENTS_ROOT="${ROST_AI_ATTACHMENTS_ROOT:-/home/rost/files/doc_attachments}"
export ROST_AI_UPLOADS_ROOT="${ROST_AI_UPLOADS_ROOT:-${ROOT_DIR}/runtime/uploads}"

mkdir -p "${ROST_AI_UPLOADS_ROOT}"

exec "${ROOT_DIR}/.venv/bin/uvicorn" app.main:app \
  --host "${ROST_AI_HOST:-127.0.0.1}" \
  --port "${ROST_AI_PORT:-18080}" \
  --workers "${ROST_AI_WORKERS:-1}"
