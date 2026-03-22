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

Environment variables:

- `ROST_AI_MODEL_DIR` - path to extracted OpenVINO model directory
- `ROST_AI_TASK` - defaults to `segment`
- `ROST_AI_IMAGE_SIZE` - defaults to `1280`
- `ROST_AI_DEFAULT_CONF` - defaults to `0.25`
- `ROST_AI_DEFAULT_IOU` - defaults to `0.45`

## API

- `GET /healthz`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`

Example:

```bash
curl -F "file=@/path/to/image.jpg" "http://127.0.0.1:8000/predict?summary=1&boxes=1"
```
