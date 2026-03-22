from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    label: str
    conf: float
    bbox: Optional[List[float]] = None
    mask: Optional[List[List[float]]] = None


class PredictResponse(BaseModel):
    detections: List[DetectionItem] = Field(default_factory=list)
    total_count: int = 0
    summary: Dict[str, int] = Field(default_factory=dict)


class BatchPredictItem(BaseModel):
    filename: str
    result: PredictResponse


class BatchPredictResponse(BaseModel):
    items: List[BatchPredictItem]


class PathPredictRequest(BaseModel):
    relative_path: str
    storage: Literal["attachments", "uploads"] = "attachments"
    source_name: Optional[str] = None
    conf: Optional[float] = None
    masks: bool = False
    boxes: bool = True
    summary: bool = True


class PathPredictResponse(PredictResponse):
    relative_path: str
    storage: str
    source_name: Optional[str] = None


class PathBatchPredictItem(BaseModel):
    relative_path: str
    storage: Literal["attachments", "uploads"] = "attachments"
    source_name: Optional[str] = None
    conf: Optional[float] = None


class PathBatchPredictResult(BaseModel):
    relative_path: str
    storage: str
    source_name: Optional[str] = None
    result: PredictResponse


class PathBatchPredictRequest(BaseModel):
    items: List[PathBatchPredictItem]
    masks: bool = False
    boxes: bool = True
    summary: bool = True


class PathBatchPredictResponse(BaseModel):
    items: List[PathBatchPredictResult]
