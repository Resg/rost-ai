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
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Recommended local port for integration with `rost`:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 18080 --reload
```

## Local Docker development

```bash
cp .env.example .env
docker compose -f docker-compose.dev.yml up --build
```

This starts the service on `http://127.0.0.1:18080` and mounts local `app/` and `models/`.

Environment variables:

- `ROST_AI_MODEL_DIR` - path to extracted OpenVINO model directory
- `ROST_AI_TASK` - defaults to `segment`
- `ROST_AI_IMAGE_SIZE` - defaults to `1280`
- `ROST_AI_DEFAULT_CONF` - defaults to `0.25`
- `ROST_AI_DEFAULT_IOU` - defaults to `0.45`

## Integration with rost

For local `rost` development point Django to the local AI service:

```bash
export AI_SERVICE_URL=http://127.0.0.1:18080
export AI_SERVICE_TIMEOUT=300
```

## API

- `GET /healthz`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`

Example:

```bash
curl -F "file=@/path/to/image.jpg" "http://127.0.0.1:8000/predict?summary=1&boxes=1"
```
