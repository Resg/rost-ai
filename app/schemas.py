from typing import Dict, List, Optional

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
