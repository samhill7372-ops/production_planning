"""Pydantic models for request/response validation."""
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


# ============== Simple Prediction ==============

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


# ============== KD Distribution Prediction ==============

class KDPredictionRequest(BaseModel):
    """Input for KD distribution prediction."""
    input_material: str = Field(..., example="10AS3BKS", description="Input KS material code")
    input_plant: str = Field(..., example="1M02", description="Plant code")
    total_input_bf: float = Field(..., example=110608, description="Total input board feet")


class GradeBreakdown(BaseModel):
    """Grade-level breakdown within a KD material."""
    grade: str = Field(..., description="Grade code (2C, 1C, PR, etc.)")
    contrib_pct: float = Field(..., description="Contribution percentage of this grade")
    expected_bf: float = Field(..., description="Expected output BF for this grade")
    orders: int = Field(..., description="Number of historical orders")
    avg_length: Optional[float] = Field(None, description="Average length in inches")
    avg_width: Optional[float] = Field(None, description="Average width in inches")
    thickness: Optional[float] = Field(None, description="Thickness in inches")


class KDOutput(BaseModel):
    """Single KD material output with grade breakdown."""
    kd_material: str = Field(..., description="KD material code")
    contrib_pct: float = Field(..., description="Contribution percentage of total output")
    expected_bf: float = Field(..., description="Expected output BF")
    orders: int = Field(..., description="Number of historical orders")
    hist_bf: float = Field(..., description="Historical total BF")
    grades: List[GradeBreakdown] = Field(default=[], description="Grade-level breakdown")


class PredictionSummary(BaseModel):
    """Summary statistics for the prediction."""
    ml_predicted_yield_pct: float = Field(..., description="ML model predicted yield percentage")
    historical_yield_pct: float = Field(..., description="Historical average yield percentage")
    difference_pct: float = Field(..., description="Difference between ML and historical")
    predicted_output_bf: float = Field(..., description="Total predicted output BF")
    ml_confidence: str = Field(..., description="ML confidence level (HIGH, MEDIUM, LOW)")
    historical_input_bf: float = Field(..., description="Total historical input BF")
    historical_output_bf: float = Field(..., description="Total historical output BF")
    historical_orders: int = Field(..., description="Number of historical orders")
    prediction_method: Optional[str] = Field(None, description="Method used for prediction")


class KDPredictionResponse(BaseModel):
    """Response with KD material distribution and grade breakdown."""
    input_material: str = Field(..., description="Input material code")
    input_plant: str = Field(..., description="Plant code")
    input_bf: float = Field(..., description="Input board feet")
    summary: PredictionSummary = Field(..., description="Prediction summary statistics")
    kd_outputs: List[KDOutput] = Field(..., description="List of KD materials with grade breakdown")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    model_version: str = Field(..., description="Model version used")


# ============== Multi-Output Prediction ==============

class TallyRecord(BaseModel):
    """Single 261 consumption tally record."""
    MANUFACTURINGORDER: str = Field(..., description="Manufacturing order number")
    MATERIALSPECIE: Optional[str] = Field(None, description="Material species (e.g. HM, SM, SPF)")
    MATERIALTHICKNESS: Optional[str] = Field(None, description="Material thickness")
    PLANT: Optional[str] = Field(None, description="Plant code")
    TALLYGRADE: Optional[str] = Field(None, description="Tally grade (e.g. 1C, 2C)")
    TALLYLENGTH: float = Field(..., description="Tally length in inches")
    TALLYWIDTH: float = Field(..., description="Tally width in inches")
    BFIN: float = Field(..., description="Board feet input")
    GOODSMOVEMENTTYPE: Optional[str] = Field("261", description="Goods movement type")

    class Config:
        json_schema_extra = {
            "example": {
                "MANUFACTURINGORDER": "1088876",
                "MATERIALSPECIE": "HM",
                "MATERIALTHICKNESS": "4",
                "PLANT": "1M02",
                "TALLYGRADE": "1C",
                "TALLYLENGTH": 144.0,
                "TALLYWIDTH": 72.0,
                "BFIN": 700.0,
                "GOODSMOVEMENTTYPE": "261"
            }
        }


class MultiOutputRequest(BaseModel):
    """Request: list of 261 tally records (one or many manufacturing orders)."""
    records: List[TallyRecord] = Field(..., min_length=1, max_length=10000, description="List of 261 consumption tally records")

    @model_validator(mode="after")
    def validate_records(self):
        if not self.records:
            raise ValueError("At least one tally record is required")
        return self


class BinOutput(BaseModel):
    """Single output bin (Grade/Length/Width combination)."""
    grade: str = Field(..., description="Output grade")
    length: str = Field(..., description="Output length")
    width: str = Field(..., description="Output width")
    boards: int = Field(..., description="Predicted board count")


class OrderResult(BaseModel):
    """Prediction result for a single manufacturing order."""
    manufacturing_order: str = Field(..., description="Manufacturing order number")
    total_bfin: float = Field(..., description="Total input board feet")
    yield_factor: float = Field(..., description="Predicted yield factor (0-1)")
    predicted_output: float = Field(..., description="Predicted output board feet")
    total_boards: int = Field(..., description="Total predicted output boards")
    distribution: List[BinOutput] = Field(..., description="Board distribution across Grade/Length/Width bins")


class MultiOutputResponse(BaseModel):
    """Response: per-order predictions + overall summary."""
    num_orders: int = Field(..., description="Number of manufacturing orders processed")
    total_bfin: float = Field(..., description="Total input board feet across all orders")
    avg_yield_factor: float = Field(..., description="Average yield factor across orders")
    total_predicted_output: float = Field(..., description="Total predicted output board feet")
    total_boards: int = Field(..., description="Total predicted boards across all orders")
    orders: List[OrderResult] = Field(..., description="Per-order prediction results")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    model_version: str = Field(..., description="Model version identifier")
    records_received: Optional[int] = Field(None, description="Total input records received")
    records_processed: Optional[int] = Field(None, description="Records after 261 movement type filter")


# ============== KD ML Distribution Prediction ==============

class KDMLPredictionItem(BaseModel):
    """Single tally record for KD ML distribution prediction.

    Records with the same manufacturing_order are grouped together and
    processed as one order (matching Streamlit CSV upload behavior).
    """
    manufacturing_order: str = Field("ORDER_0001", description="Manufacturing order number. Records with the same value are grouped as one order.")
    material: str = Field(..., description="Input KS material code")
    species: str = Field(..., description="Wood species code")
    thickness: float = Field(..., description="Material thickness in inches")
    plant: str = Field(..., description="Plant code")
    tallygrade: str = Field(..., description="Tally grade")
    tallylength: float = Field(..., description="Tally length in inches")
    tallywidth: float = Field(..., description="Tally width in inches")
    bfin: float = Field(..., gt=0, description="Board feet input")
    posting_date: Optional[str] = Field(None, description="Posting date (metadata only, not used in prediction)")

    class Config:
        json_schema_extra = {
            "example": {
                "manufacturing_order": "1078021",
                "material": "4RO3BKS",
                "species": "RO",
                "thickness": 4,
                "plant": "1M10",
                "tallygrade": "PR",
                "tallylength": 119,
                "tallywidth": 72,
                "bfin": 2255,
                "posting_date": "2025-06-15"
            }
        }


class KDMLPredictionRequest(BaseModel):
    """Bulk prediction request: array of tally records.

    Records are grouped by manufacturing_order. Each group produces one
    prediction result, using all tally rows together for feature engineering
    (matching how the Streamlit CSV upload works).
    """
    items: List[KDMLPredictionItem] = Field(..., min_length=1, max_length=10000, description="List of tally records for prediction")

    @model_validator(mode="after")
    def validate_items(self):
        if not self.items:
            raise ValueError("At least one tally record is required")
        return self


class KDMLOutputMaterial(BaseModel):
    """Single output material in the KD distribution — V2.1 simplified."""
    material: str = Field(..., description="KD output material code")
    bfout: float = Field(..., description="Expected output board feet")


class KDMLOrderResult(BaseModel):
    """Prediction result for one manufacturing order."""
    manufacturing_order: str = Field(..., description="Manufacturing order number")
    material: str = Field(..., description="Input KS material code")
    total_bfin: float = Field(..., description="Total input board feet (summed across all tally rows)")
    num_tally_rows: int = Field(..., description="Number of tally rows in this order")
    predicted_yield_pct: float = Field(..., description="Predicted yield percentage")
    predicted_output_bf: float = Field(..., description="Predicted output board feet")
    search_mode: str = Field(..., description="KNN search mode used")
    n_neighbors_used: int = Field(..., description="Number of KNN neighbors used")
    avg_neighbor_distance: float = Field(..., description="Average KNN neighbor distance")
    kd_count: int = Field(..., description="Number of KD materials in distribution")
    distribution: List[KDMLOutputMaterial] = Field(..., description="KD material distribution")


class KDMLErrorItem(BaseModel):
    """Error details for a failed order prediction."""
    manufacturing_order: str = Field(..., description="Manufacturing order number")
    error: str = Field(..., description="Error message")


class KDMLPredictionResponse(BaseModel):
    """Bulk KD ML prediction response — one result per manufacturing order."""
    results: List[KDMLOrderResult] = Field(..., description="Per-order prediction results")
    num_orders: int = Field(..., description="Number of manufacturing orders processed")
    total_records: int = Field(..., description="Total tally records received")
    successful: int = Field(..., description="Number of successful order predictions")
    failed: int = Field(..., description="Number of failed order predictions")
    errors: List[KDMLErrorItem] = Field(default_factory=list, description="Error details for failed orders")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    model_version: str = Field(..., description="Model version identifier")
