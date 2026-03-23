from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import os

from PIL import Image, ImageOps
from ultralytics import YOLO

from app.config import get_settings


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(__import__("io").BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _load_image_from_path(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _resolve_root(storage: str) -> Path:
    settings = get_settings()
    if storage == "attachments":
        return Path(settings.attachments_root)
    if storage == "uploads":
        return Path(settings.uploads_root)
    raise ValueError("Unsupported storage: %s" % storage)


def resolve_storage_path(relative_path: str, storage: str = "attachments") -> Path:
    if not relative_path:
        raise ValueError("relative_path is required")
    root = _resolve_root(storage).resolve()
    normalized = os.path.normpath(relative_path).lstrip(os.sep)
    path = (root / normalized).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Resolved path escapes storage root")
    return path


def normalize_model_code(model_code: str = None) -> str:
    settings = get_settings()
    normalized = (model_code or settings.default_model_code or "").strip()
    if not normalized:
        raise RuntimeError("Model code is not configured.")
    return normalized


def resolve_model_dir(model_code: str = None) -> Path:
    settings = get_settings()
    resolved_model_code = normalize_model_code(model_code)
    candidates = []

    candidates.append(Path(settings.models_root) / resolved_model_code)
    candidates.append(Path(settings.models_root) / resolved_model_code / "best_openvino_model")

    if resolved_model_code == settings.default_model_code:
        candidates.append(Path(settings.model_dir))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Model directory does not exist for %s. Tried: %s" % (
            resolved_model_code,
            ", ".join([str(candidate) for candidate in candidates]),
        )
    )


@lru_cache(maxsize=16)
def get_model(model_code: str = None) -> YOLO:
    settings = get_settings()
    model_dir = resolve_model_dir(model_code)
    if not model_dir.exists():
        raise RuntimeError("Model directory does not exist: %s" % model_dir)
    return YOLO(str(model_dir), task=settings.task)


def get_model_info(model_code: str = None) -> Dict[str, Any]:
    settings = get_settings()
    resolved_model_code = normalize_model_code(model_code)
    model_dir = resolve_model_dir(resolved_model_code)
    model = get_model(resolved_model_code)
    return {
        "model_code": resolved_model_code,
        "default_model_code": settings.default_model_code,
        "model_dir": str(model_dir),
        "attachments_root": str(settings.attachments_root),
        "uploads_root": str(settings.uploads_root),
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
    model_code: str = None,
) -> Dict[str, Any]:
    image = _load_image(image_bytes)
    return _predict_loaded_image(
        image,
        include_masks=include_masks,
        include_boxes=include_boxes,
        include_summary=include_summary,
        conf=conf,
        model_code=model_code,
    )


def _predict_loaded_image(
    image: Image.Image,
    include_masks: bool = False,
    include_boxes: bool = True,
    include_summary: bool = True,
    conf: float = None,
    model_code: str = None,
) -> Dict[str, Any]:
    settings = get_settings()
    resolved_model_code = normalize_model_code(model_code)
    model = get_model(resolved_model_code)
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
    response["model_code"] = resolved_model_code
    response["image_width"] = int(image.width or 0)
    response["image_height"] = int(image.height or 0)

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


def predict_relative_path(
    relative_path: str,
    storage: str = "attachments",
    include_masks: bool = False,
    include_boxes: bool = True,
    include_summary: bool = True,
    conf: float = None,
    model_code: str = None,
) -> Dict[str, Any]:
    image_path = resolve_storage_path(relative_path, storage=storage)
    if not image_path.exists():
        raise FileNotFoundError("Image path does not exist: %s" % image_path)
    image = _load_image_from_path(image_path)
    return _predict_loaded_image(
        image,
        include_masks=include_masks,
        include_boxes=include_boxes,
        include_summary=include_summary,
        conf=conf,
        model_code=model_code,
    )
