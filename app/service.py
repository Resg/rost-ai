from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import os
import queue
import threading
import time

from PIL import Image, ImageOps
from ultralytics import YOLO
import yaml

from app.config import get_settings


class InferenceQueueFull(RuntimeError):
    pass


class InferenceTimeout(RuntimeError):
    pass


class _InferenceJob(object):
    def __init__(
        self,
        job_type: str,
        model_code: str,
        include_masks: bool,
        include_boxes: bool,
        include_summary: bool,
        conf: float = None,
        image_bytes: bytes = None,
        image_path: str = None,
    ) -> None:
        self.job_type = job_type
        self.model_code = model_code
        self.include_masks = include_masks
        self.include_boxes = include_boxes
        self.include_summary = include_summary
        self.conf = conf
        self.image_bytes = image_bytes
        self.image_path = image_path
        self.result = None
        self.error = None
        self.done = threading.Event()
        self.queued_at = time.time()


_inference_queue = None
_inference_queue_lock = threading.Lock()
_inference_threads = []
_active_jobs = 0
_active_jobs_lock = threading.Lock()


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


def _ensure_openvino_runtime_dir(candidate: Path) -> Path:
    candidate = Path(candidate)
    if not candidate.is_dir():
        return candidate

    best_xml = candidate / "best.xml"
    if not best_xml.exists():
        return candidate

    if candidate.name.endswith("_openvino_model"):
        return candidate

    settings = get_settings()
    runtime_root = Path(settings.uploads_root) / "model_aliases"
    runtime_root.mkdir(parents=True, exist_ok=True)
    alias_dir = runtime_root / ("%s_openvino_model" % candidate.name)

    try:
        if alias_dir.is_symlink():
            if alias_dir.resolve() == candidate.resolve():
                return alias_dir
            alias_dir.unlink()
        elif alias_dir.exists():
            return alias_dir

        os.symlink(str(candidate.resolve()), str(alias_dir))
        return alias_dir
    except OSError:
        return candidate


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


def _resolve_model_entry(candidate: Path) -> Path:
    candidate = Path(candidate)
    if not candidate.exists():
        return None

    if candidate.is_file():
        return candidate

    candidate = _ensure_openvino_runtime_dir(candidate)

    best_xml = candidate / "best.xml"
    if best_xml.exists():
        return candidate

    xml_files = sorted(candidate.glob("*.xml"))
    if len(xml_files) == 1:
        return xml_files[0]

    return candidate


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
        resolved_entry = _resolve_model_entry(candidate)
        if resolved_entry is not None:
            return resolved_entry

    raise RuntimeError(
        "Model directory does not exist for %s. Tried: %s" % (
            resolved_model_code,
            ", ".join([str(candidate) for candidate in candidates]),
        )
    )


@lru_cache(maxsize=16)
def get_model(model_code: str = None) -> YOLO:
    model_dir = resolve_model_dir(model_code)
    if not model_dir.exists():
        raise RuntimeError("Model directory does not exist: %s" % model_dir)
    return YOLO(str(model_dir), task=get_model_task(model_code))


def _create_model(model_code: str = None) -> YOLO:
    model_dir = resolve_model_dir(model_code)
    if not model_dir.exists():
        raise RuntimeError("Model directory does not exist: %s" % model_dir)
    return YOLO(str(model_dir), task=get_model_task(model_code))


@lru_cache(maxsize=16)
def get_model_metadata(model_code: str = None) -> Dict[str, Any]:
    model_dir = resolve_model_dir(model_code)
    metadata_root = model_dir.parent if model_dir.is_file() else model_dir
    metadata_path = metadata_root / "metadata.yaml"
    if not metadata_path.exists():
        raise RuntimeError("Model metadata does not exist: %s" % metadata_path)

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return yaml.safe_load(metadata_file) or {}


@lru_cache(maxsize=16)
def get_model_labels(model_code: str = None) -> Dict[int, str]:
    payload = get_model_metadata(model_code)

    names = payload.get("names") or {}
    labels = {}

    if isinstance(names, dict):
        for key, value in names.items():
            labels[int(key)] = str(value)
        return labels

    if isinstance(names, (list, tuple)):
        for index, value in enumerate(names):
            labels[index] = str(value)
        return labels

    raise RuntimeError("Model metadata does not contain valid labels.")


def get_model_task(model_code: str = None) -> str:
    settings = get_settings()
    metadata = get_model_metadata(model_code)
    return str(metadata.get("task") or settings.task)


def get_model_image_size(model_code: str = None) -> int:
    settings = get_settings()
    metadata = get_model_metadata(model_code)
    raw_imgsz = metadata.get("imgsz")

    if isinstance(raw_imgsz, (list, tuple)) and raw_imgsz:
        try:
            return int(raw_imgsz[0])
        except (TypeError, ValueError):
            return int(settings.image_size)

    if raw_imgsz not in (None, ""):
        try:
            return int(raw_imgsz)
        except (TypeError, ValueError):
            return int(settings.image_size)

    return int(settings.image_size)


def get_model_info(model_code: str = None) -> Dict[str, Any]:
    settings = get_settings()
    resolved_model_code = normalize_model_code(model_code)
    model_dir = resolve_model_dir(resolved_model_code)
    get_model(resolved_model_code)
    labels = get_model_labels(resolved_model_code)
    task = get_model_task(resolved_model_code)
    image_size = get_model_image_size(resolved_model_code)
    return {
        "model_code": resolved_model_code,
        "default_model_code": settings.default_model_code,
        "model_dir": str(model_dir),
        "attachments_root": str(settings.attachments_root),
        "uploads_root": str(settings.uploads_root),
        "task": task,
        "image_size": image_size,
        "inference_workers": settings.inference_workers,
        "inference_queue_size": settings.inference_queue_size,
        "labels": labels,
    }


def get_health_status() -> Dict[str, Any]:
    settings = get_settings()
    inference_queue = _ensure_inference_pool()
    with _active_jobs_lock:
        active_jobs = _active_jobs
    return {
        "status": "ok",
        "inference_workers": settings.inference_workers,
        "queue_size": inference_queue.qsize(),
        "queue_capacity": settings.inference_queue_size,
        "active_jobs": active_jobs,
    }


def predict_image(
    image_bytes: bytes,
    include_masks: bool = False,
    include_boxes: bool = True,
    include_summary: bool = True,
    conf: float = None,
    model_code: str = None,
) -> Dict[str, Any]:
    job = _InferenceJob(
        job_type="bytes",
        model_code=model_code,
        include_masks=include_masks,
        include_boxes=include_boxes,
        include_summary=include_summary,
        conf=conf,
        image_bytes=image_bytes,
    )
    return _run_inference_job(job)


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
    labels = get_model_labels(resolved_model_code)
    image_size = get_model_image_size(resolved_model_code)
    threshold = settings.default_conf if conf is None else conf
    result = model.predict(
        image,
        imgsz=image_size,
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
            label = labels.get(int(box.cls[0]), str(int(box.cls[0])))
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


def _predict_loaded_image_with_model(
    image: Image.Image,
    model: YOLO,
    settings,
    include_masks: bool = False,
    include_boxes: bool = True,
    include_summary: bool = True,
    conf: float = None,
    model_code: str = None,
) -> Dict[str, Any]:
    resolved_model_code = normalize_model_code(model_code)
    labels = get_model_labels(resolved_model_code)
    image_size = get_model_image_size(resolved_model_code)
    threshold = settings.default_conf if conf is None else conf
    result = model.predict(
        image,
        imgsz=image_size,
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
            label = labels.get(int(box.cls[0]), str(int(box.cls[0])))
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


def _ensure_inference_pool():
    global _inference_queue
    if _inference_queue is not None:
        return _inference_queue

    with _inference_queue_lock:
        if _inference_queue is not None:
            return _inference_queue

        settings = get_settings()
        _inference_queue = queue.Queue(maxsize=max(1, int(settings.inference_queue_size)))
        for index in range(max(1, int(settings.inference_workers))):
            thread = threading.Thread(
                target=_inference_worker_loop,
                args=(index + 1,),
                name="inference-worker-%s" % (index + 1),
            )
            thread.daemon = True
            thread.start()
            _inference_threads.append(thread)
        return _inference_queue


def _inference_worker_loop(worker_id: int) -> None:
    settings = get_settings()
    worker_models = {}
    inference_queue = _inference_queue

    while True:
        job = inference_queue.get()
        with _active_jobs_lock:
            global _active_jobs
            _active_jobs += 1
        try:
            resolved_model_code = normalize_model_code(job.model_code)
            model = worker_models.get(resolved_model_code)
            if model is None:
                model = _create_model(resolved_model_code)
                worker_models[resolved_model_code] = model

            if job.job_type == "bytes":
                image = _load_image(job.image_bytes)
            else:
                image = _load_image_from_path(Path(job.image_path))

            job.result = _predict_loaded_image_with_model(
                image,
                model=model,
                settings=settings,
                include_masks=job.include_masks,
                include_boxes=job.include_boxes,
                include_summary=job.include_summary,
                conf=job.conf,
                model_code=resolved_model_code,
            )
        except Exception as exc:
            job.error = exc
        finally:
            job.done.set()
            inference_queue.task_done()
            with _active_jobs_lock:
                _active_jobs -= 1


def _run_inference_job(job: _InferenceJob) -> Dict[str, Any]:
    settings = get_settings()
    inference_queue = _ensure_inference_pool()

    try:
        inference_queue.put(job, block=False)
    except queue.Full:
        raise InferenceQueueFull("Inference queue is full. Try again later.")

    if not job.done.wait(timeout=max(1, int(settings.inference_wait_timeout))):
        raise InferenceTimeout("Inference timed out while waiting in queue.")

    if job.error is not None:
        raise job.error

    return job.result


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
    job = _InferenceJob(
        job_type="path",
        model_code=model_code,
        include_masks=include_masks,
        include_boxes=include_boxes,
        include_summary=include_summary,
        conf=conf,
        image_path=str(image_path),
    )
    return _run_inference_job(job)
