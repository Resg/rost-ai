from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageOps
from ultralytics import YOLO

from app.config import get_settings


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(__import__("io").BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    if not model_dir.exists():
        raise RuntimeError("Model directory does not exist: %s" % model_dir)
    return YOLO(str(model_dir), task=settings.task)


def get_model_info() -> Dict[str, Any]:
    settings = get_settings()
    model = get_model()
    return {
        "model_dir": str(settings.model_dir),
        "task": settings.task,
        "image_size": settings.image_size,
        "labels": model.names,
    }


def predict_image(
    image_bytes: bytes,
    include_masks: bool = False,
    include_boxes: bool = True,
    include_summary: bool = True,
    conf: float = None,
) -> Dict[str, Any]:
    settings = get_settings()
    model = get_model()
    image = _load_image(image_bytes)
    threshold = settings.default_conf if conf is None else conf
    result = model.predict(
        image,
        imgsz=settings.image_size,
        conf=threshold,
        iou=settings.default_iou,
        verbose=False,
    )[0]

    response: Dict[str, Any] = {}
    found_labels: List[str] = []
    detections: List[Dict[str, Any]] = []

    if result.boxes is not None and len(result.boxes):
        for index in range(len(result.boxes)):
            box = result.boxes[index]
            label = model.names[int(box.cls[0])]
            found_labels.append(label)
            item = {
                "label": label,
                "conf": round(float(box.conf[0]), 3),
            }
            if include_boxes:
                item["bbox"] = [float(value) for value in box.xyxy[0].tolist()]
            if include_masks and result.masks is not None:
                polygon = result.masks.xy[index]
                item["mask"] = [[float(x), float(y)] for x, y in polygon.tolist()]
            detections.append(item)

    response["detections"] = detections
    if include_summary:
        response["total_count"] = len(found_labels)
        response["summary"] = dict(Counter(found_labels))
    return response
