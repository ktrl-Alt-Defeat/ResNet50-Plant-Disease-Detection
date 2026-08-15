from typing import List
from pydantic import BaseModel, Field, ConfigDict


class APIInfoResponse(BaseModel):
    """API metadata schema."""
    name: str = Field(..., example="ResNet-50 Crop Disease Classification API")
    model_name: str = Field(..., example="ResNet-50")
    num_classes: int = Field(..., example=38)
    status: str = Field(..., example="online")


class HealthResponse(BaseModel):
    """Application health check response schema."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cuda:0")


class PredictionItem(BaseModel):
    """Individual class prediction item."""
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class", serialization_alias="class", example="tomato___target_spot")
    confidence: float = Field(..., example=0.9842)


class PredictionResponse(BaseModel):
    """Complete prediction response containing top-1 and top-K results."""
    predicted_class: str = Field(..., example="tomato___target_spot")
    confidence: float = Field(..., example=0.9842)
    confidence_percentage: float = Field(..., example=98.42)
    top_5_predictions: List[PredictionItem]


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    detail: str = Field(..., example="Invalid image file provided.")
    error_type: str = Field(..., example="BadRequestError")
    status_code: int = Field(..., example=400)
