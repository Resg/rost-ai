from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config import get_settings
from app.schemas import (
    BatchPredictItem,
    BatchPredictResponse,
    PathBatchPredictRequest,
    PathBatchPredictResponse,
    PathBatchPredictResult,
    PathPredictRequest,
    PathPredictResponse,
    PredictResponse,
)
from app.service import get_model_info, predict_image, predict_relative_path

settings = get_settings()
app = FastAPI(title=settings.app_name)


def _run_path_prediction(relative_path, storage, masks, boxes, summary, conf, model_code=None):
    try:
        return predict_relative_path(
            relative_path,
            storage=storage,
            include_masks=masks,
            include_boxes=boxes,
            include_summary=summary,
            conf=conf,
            model_code=model_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/model/info")
def model_info(model_code: str = Query(None)):
    return get_model_info(model_code=model_code)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    masks: bool = Query(False),
    boxes: bool = Query(True),
    summary: bool = Query(True),
    conf: float = Query(None),
    model_code: str = Query(None),
):
    contents = await file.read()
    return predict_image(
        contents,
        include_masks=masks,
        include_boxes=boxes,
        include_summary=summary,
        conf=conf,
        model_code=model_code,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    masks: bool = Query(False),
    boxes: bool = Query(True),
    summary: bool = Query(True),
    conf: float = Query(None),
    model_code: str = Query(None),
):
    items = []
    for upload in files:
        contents = await upload.read()
        items.append(
            BatchPredictItem(
                filename=upload.filename,
                result=predict_image(
                    contents,
                    include_masks=masks,
                    include_boxes=boxes,
                    include_summary=summary,
                    conf=conf,
                    model_code=model_code,
                ),
            )
        )
    return BatchPredictResponse(items=items)


@app.post("/predict/path", response_model=PathPredictResponse)
def predict_path(payload: PathPredictRequest):
    result = _run_path_prediction(
        payload.relative_path,
        payload.storage,
        payload.masks,
        payload.boxes,
        payload.summary,
        payload.conf,
        payload.model_code,
    )
    return PathPredictResponse(
        relative_path=payload.relative_path,
        storage=payload.storage,
        source_name=payload.source_name,
        **result
    )


@app.post("/predict/paths", response_model=PathBatchPredictResponse)
def predict_paths(payload: PathBatchPredictRequest):
    items = []
    for item in payload.items:
        result = _run_path_prediction(
            item.relative_path,
            item.storage,
            payload.masks,
            payload.boxes,
            payload.summary,
            item.conf,
            item.model_code,
        )
        items.append(
            PathBatchPredictResult(
                relative_path=item.relative_path,
                storage=item.storage,
                source_name=item.source_name,
                result=PredictResponse(**result),
            )
        )
    return PathBatchPredictResponse(items=items)
