"""FastAPI application with ONNX model loaded at startup."""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictionRequest, PredictionResponse, HealthResponse, ErrorResponse,
    KDPredictionRequest, KDPredictionResponse
)
from inference import ONNXModelService
from preprocessing import Preprocessor
from kd_service import KDDistributionService

# Global instances (loaded once at startup)
model_service: ONNXModelService = None
preprocessor: Preprocessor = None
kd_service: KDDistributionService = None
model_version: str = None

# Confidence margin (2 * std from model metrics, ~3.94 RMSE)
CONFIDENCE_MARGIN = 7.88


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and encoders ONCE at startup."""
    global model_service, preprocessor, kd_service, model_version

    # Support both MODEL_NAME (new multi-service) and MODEL_YEAR (legacy)
    model_version = os.getenv("MODEL_NAME") or os.getenv("MODEL_YEAR", "2024")

    # For legacy MODEL_YEAR like "2024", map to model folder name
    # This allows both MODEL_YEAR=2024 and MODEL_NAME=yield-2024 to work
    model_folder = model_version

    # Support both Docker (/app/models) and local (./models) paths
    if os.path.exists(f"/app/models/{model_folder}"):
        model_dir = f"/app/models/{model_folder}"
    else:
        # Local development path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models", model_folder)

    print(f"Loading model from {model_dir}...")
    model_service = ONNXModelService(f"{model_dir}/yield_model.onnx")
    preprocessor = Preprocessor(model_dir)

    # Load KD distribution service with ML model for yield prediction
    try:
        kd_service = KDDistributionService(
            model_dir,
            model_service=model_service,
            preprocessor=preprocessor
        )
        print("KD Distribution service loaded successfully (with ML model)!")
    except FileNotFoundError as e:
        print(f"Warning: KD service not available - {e}")
        kd_service = None

    print(f"Model {model_version} loaded successfully!")

    yield  # App runs here

    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="Yield Prediction API",
    description="Production ML inference service for material yield prediction",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# CORS (adjust origins for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict yield percentage and output board feet.

    Accepts raw material values (not encoded).
    Returns prediction with confidence bounds and latency.
    """
    start_time = time.perf_counter()

    try:
        # Encode raw values to features
        features = preprocessor.encode(request)

        # Run inference
        yield_pct = model_service.predict(features)

        # Calculate output
        output_bf = request.total_input_bf * (yield_pct / 100)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            predicted_yield_pct=round(yield_pct, 2),
            predicted_output_bf=round(output_bf, 2),
            confidence_lower=round(max(0, yield_pct - CONFIDENCE_MARGIN), 2),
            confidence_upper=round(min(150, yield_pct + CONFIDENCE_MARGIN), 2),
            latency_ms=round(latency_ms, 2),
            model_version=model_version
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint for container orchestration."""
    return HealthResponse(
        status="healthy" if model_service else "unhealthy",
        model_loaded=model_service is not None,
        model_version=model_version or "not loaded"
    )


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Yield Prediction API",
        "version": "1.0.0",
        "model_version": model_version,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/valid-values/{field}", tags=["Info"])
async def get_valid_values(field: str):
    """Get valid values for a categorical field."""
    field_map = {
        "input_material": "Input_Material",
        "input_specie": "Input_Specie",
        "input_grade": "Input_Grade",
        "input_plant": "Input_Plant",
    }

    if field not in field_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid fields: {list(field_map.keys())}"
        )

    values = preprocessor.get_valid_values(field_map[field])
    return {"field": field, "valid_values": values, "count": len(values)}


# ============== KD Distribution Prediction ==============

@app.post("/predict-kd", response_model=KDPredictionResponse, tags=["KD Prediction"])
async def predict_kd(request: KDPredictionRequest):
    """
    Predict KD material distribution with grade breakdown.

    Returns multiple KD output materials with:
    - Contribution percentages
    - Expected output BF
    - Grade-level breakdown (2C, 1C, PR, etc.)
    - Dimensional details (length, width, thickness)

    This matches the Streamlit app's KD Material Distribution table.
    """
    start_time = time.perf_counter()

    if kd_service is None:
        raise HTTPException(
            status_code=503,
            detail="KD Distribution service not available. Missing historical_data_precomputed.joblib"
        )

    try:
        # Get KD distribution prediction
        result = kd_service.get_kd_prediction(
            input_material=request.input_material,
            plant=request.input_plant,
            input_bf=request.total_input_bf
        )

        # Check for error
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return KDPredictionResponse(
            input_material=result['input_material'],
            input_plant=result['input_plant'],
            input_bf=result['input_bf'],
            summary=result['summary'],
            kd_outputs=result['kd_outputs'],
            latency_ms=round(latency_ms, 2),
            model_version=model_version
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KD prediction error: {str(e)}")


@app.get("/kd-materials", tags=["KD Prediction"])
async def list_kd_materials():
    """List available KS materials for KD prediction."""
    if kd_service is None:
        raise HTTPException(
            status_code=503,
            detail="KD Distribution service not available"
        )

    materials = kd_service.get_available_materials()
    return {"materials": materials, "count": len(materials)}


@app.get("/kd-plants/{material}", tags=["KD Prediction"])
async def list_kd_plants(material: str):
    """List available plants for a specific KS material."""
    if kd_service is None:
        raise HTTPException(
            status_code=503,
            detail="KD Distribution service not available"
        )

    plants = kd_service.get_available_plants(material)
    if not plants:
        raise HTTPException(
            status_code=404,
            detail=f"Material not found: {material}"
        )

    return {"material": material, "plants": plants, "count": len(plants)}
