from functools import lru_cache
from pathlib import Path

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    app_name: str = "rost-ai"
    model_dir: Path = Path("models/best_openvino_model")
    models_root: Path = Path("models")
    default_model_code: str = "nanofarma-openvino-v1"
    attachments_root: Path = Path("/home/rost/files/doc_attachments")
    uploads_root: Path = Path("runtime/uploads")
    task: str = "segment"
    # Cabinet images are already downscaled on upload; 640 keeps recall higher
    # on shelf photos than the larger default.
    image_size: int = 640
    default_conf: float = 0.25
    default_iou: float = 0.45
    inference_workers: int = 2
    inference_queue_size: int = 32
    inference_wait_timeout: int = 300

    class Config:
        env_prefix = "ROST_AI_"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
