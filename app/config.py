from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "rost-ai"
    model_dir: Path = Path("models/best_openvino_model")
    task: str = "segment"
    image_size: int = 1280
    default_conf: float = 0.25
    default_iou: float = 0.45

    class Config:
        env_prefix = "ROST_AI_"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
