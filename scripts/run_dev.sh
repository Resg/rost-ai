#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROST_REPO_DIR="${ROOT_DIR}/../rost"

DEFAULT_ATTACHMENTS_ROOT="${ROOT_DIR}/runtime/attachments"
DEFAULT_UPLOADS_ROOT="${ROOT_DIR}/runtime/uploads"
if [ -d "${ROST_REPO_DIR}/files/doc_attachments" ]; then
  DEFAULT_ATTACHMENTS_ROOT="${ROST_REPO_DIR}/files/doc_attachments"
fi
if [ -d "${ROST_REPO_DIR}/files/media" ]; then
  DEFAULT_UPLOADS_ROOT="${ROST_REPO_DIR}/files/media/ai_uploads"
fi

if [ ! -x "${ROOT_DIR}/.venv/bin/uvicorn" ]; then
  "${ROOT_DIR}/scripts/bootstrap_venv.sh"
fi

export ROST_AI_MODEL_DIR="${ROST_AI_MODEL_DIR:-${ROOT_DIR}/models/best_openvino_model}"
export ROST_AI_ATTACHMENTS_ROOT="${ROST_AI_ATTACHMENTS_ROOT:-${DEFAULT_ATTACHMENTS_ROOT}}"
export ROST_AI_UPLOADS_ROOT="${ROST_AI_UPLOADS_ROOT:-${DEFAULT_UPLOADS_ROOT}}"

mkdir -p "${ROST_AI_UPLOADS_ROOT}"

exec "${ROOT_DIR}/.venv/bin/uvicorn" app.main:app \
  --host "${ROST_AI_HOST:-127.0.0.1}" \
  --port "${ROST_AI_PORT:-18080}" \
  --reload
