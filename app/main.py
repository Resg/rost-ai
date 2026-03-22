from typing import List

from fastapi import FastAPI, File, Query, UploadFile

from app.config import get_settings
from app.schemas import BatchPredictItem, BatchPredictResponse, PredictResponse
from app.service import get_model_info, predict_image

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    return get_model_info()


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    masks: bool = Query(False),
    boxes: bool = Query(True),
    summary: bool = Query(True),
    conf: float = Query(None),
):
    contents = await file.read()
    return predict_image(
        contents,
        include_masks=masks,
        include_boxes=boxes,
        include_summary=summary,
        conf=conf,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    masks: bool = Query(False),
    boxes: bool = Query(True),
    summary: bool = Query(True),
    conf: float = Query(None),
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
                ),
            )
        )
    return BatchPredictResponse(items=items)
