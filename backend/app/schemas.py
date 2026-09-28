from pydantic import BaseModel, Field


class Prediction(BaseModel):
    breed: str
    confidence: float = Field(ge=0, le=1)


class ImageResult(BaseModel):
    filename: str
    success: bool
    predictions: list[Prediction] = []
    animal_detected: bool | None = None
    detection_confidence: float | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    results: list[ImageResult]
