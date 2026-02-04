"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input for yield prediction - accepts raw values."""
    input_material: str = Field(..., example="6SM2CKD", description="Input material code")
    input_specie: str = Field(..., example="SPF", description="Wood specie")
    input_grade: str = Field(..., example="2&BTR", description="Material grade")
    input_plant: str = Field(..., example="0022", description="Plant code")
    input_thickness: float = Field(..., example=4.0, description="Thickness in inches")
    input_length: float = Field(..., example=96.0, description="Length in inches")
    input_width: float = Field(..., example=8.0, description="Width in inches")
    total_input_bf: float = Field(..., example=1000.0, description="Total input board feet")

    class Config:
        json_schema_extra = {
            "example": {
                "input_material": "6SM2CKD",
                "input_specie": "SPF",
                "input_grade": "2&BTR",
                "input_plant": "0022",
                "input_thickness": 4.0,
                "input_length": 96.0,
                "input_width": 8.0,
                "total_input_bf": 1000.0
            }
        }


class PredictionResponse(BaseModel):
    """Stable JSON response contract."""
    predicted_yield_pct: float = Field(..., description="Predicted yield percentage")
    predicted_output_bf: float = Field(..., description="Predicted output board feet")
    confidence_lower: float = Field(..., description="Lower bound of confidence interval")
    confidence_upper: float = Field(..., description="Upper bound of confidence interval")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Loaded model version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str = Field(..., description="Error message")
