# rost-ai

Internal inference service for `rost`.

## Model

Place the extracted OpenVINO model under:

`models/best_openvino_model/`

Expected files:

- `best.xml`
- `best.bin`
- `metadata.yaml`

The model directory is ignored by git on purpose.

## Local run

```bash
export PYTHON_BIN=python3
./scripts/bootstrap_venv.sh
./scripts/run_dev.sh
```

Recommended local port for integration with `rost`:

```bash
ROST_AI_PORT=18080 ./scripts/run_dev.sh
```

Environment variables:

- `ROST_AI_MODEL_DIR` - path to extracted OpenVINO model directory
- `ROST_AI_ATTACHMENTS_ROOT` - root with report attachments on disk
- `ROST_AI_UPLOADS_ROOT` - root for temporary uploads from `rost`
- `ROST_AI_TASK` - defaults to `segment`
- `ROST_AI_IMAGE_SIZE` - defaults to `1280`
- `ROST_AI_DEFAULT_CONF` - defaults to `0.25`
- `ROST_AI_DEFAULT_IOU` - defaults to `0.45`

## Optional Docker development

```bash
cp .env.example .env
docker compose -f docker-compose.dev.yml up --build
```

This starts the service on `http://127.0.0.1:18080` and mounts local `app/` and `models/`.

## Integration with rost

For local `rost` development point Django to the local AI service:

```bash
export AI_SERVICE_URL=http://127.0.0.1:18080
export AI_SERVICE_TIMEOUT=300
```

`rost` should call `rost-ai` with the attachment relative path. The file itself is read by `rost-ai` from `ROST_AI_ATTACHMENTS_ROOT`.

## Supervisor

Server-side direct run is prepared via:

- `scripts/run_prod.sh`
- `deploy/supervisor/rost-ai.conf`

The service is expected to run under a dedicated Python venv in `.venv` and read attachments from the shared disk.
If the host default `python3` is unsuitable, set `PYTHON_BIN` before `bootstrap_venv.sh`.

## API

- `GET /healthz`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`
- `POST /predict/path`
- `POST /predict/paths`

Example:

```bash
curl -X POST "http://127.0.0.1:18080/predict/path" \
  -H "Content-Type: application/json" \
  -d '{
    "relative_path": "44712_1_photo.jpg",
    "storage": "attachments",
    "summary": true,
    "boxes": true
  }'
```
