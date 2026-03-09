"""FastAPI application — loads ALL model years at startup."""
import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from auth import verify_credentials

from schemas import (
    PredictionRequest, PredictionResponse, HealthResponse, ErrorResponse,
    KDPredictionRequest, KDPredictionResponse,
    MultiOutputRequest, MultiOutputResponse,
    KDMLPredictionRequest, KDMLPredictionResponse,
)
from inference import ONNXModelService
from preprocessing import Preprocessor
from kd_service import KDDistributionService
from multi_output_service import MultiOutputService
from kd_ml_service import KDMLDistributionService

# ─── Global state (loaded once at startup) ────────────────────────────
# One entry per model year: {"2024": ..., "2025": ..., "2year": ...}
model_services: dict = {}
preprocessors: dict = {}
kd_services: dict = {}
available_models: list = []

# Multi-output is a single model (not year-specific)
mo_service: MultiOutputService = None

# KD ML distribution model (KNN + XGBoost, not year-specific)
kd_ml_service: KDMLDistributionService = None

# Model years to attempt loading
MODEL_YEARS = ["2024", "2025", "2year"]

# Confidence margin (2 * std from model metrics, ~3.94 RMSE)
CONFIDENCE_MARGIN = 7.88


def _resolve_model_dir(model_folder: str) -> str | None:
    """Find model directory in Docker or local paths. Returns None if not found."""
    # Docker path
    docker_path = f"/app/models/{model_folder}"
    if os.path.exists(docker_path):
        return docker_path
    # Local development path (deployment/models/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_path = os.path.join(base_dir, "models", model_folder)
    if os.path.exists(local_path):
        return local_path
    # Project root models/ directory (for models not copied to deployment/)
    project_root = os.path.dirname(base_dir)
    root_path = os.path.join(project_root, "models", model_folder)
    if os.path.exists(root_path):
        return root_path
    return None


def _get_model_or_raise(model_year: str, service_dict: dict, service_name: str):
    """Look up a model year in a service dict, raise 400 if not found."""
    if model_year not in service_dict:
        if model_year not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_year '{model_year}'. Available: {available_models}",
            )
        raise HTTPException(
            status_code=503,
            detail=f"{service_name} not available for model_year '{model_year}'",
        )
    return service_dict[model_year]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ALL model years and multi-output model at startup."""
    global mo_service, kd_ml_service

    # Load each year model
    for year in MODEL_YEARS:
        model_dir = _resolve_model_dir(year)
        if model_dir is None:
            print(f"Skipping model year {year} — directory not found")
            continue

        onnx_path = os.path.join(model_dir, "yield_model.onnx")
        if not os.path.exists(onnx_path):
            print(f"Skipping model year {year} — yield_model.onnx not found")
            continue

        try:
            ms = ONNXModelService(onnx_path)
            pp = Preprocessor(model_dir)
            model_services[year] = ms
            preprocessors[year] = pp
            available_models.append(year)
            print(f"Model {year} loaded successfully!")

            # KD service (optional — needs historical_data_precomputed.joblib)
            try:
                kd = KDDistributionService(model_dir, model_service=ms, preprocessor=pp)
                kd_services[year] = kd
                print(f"  KD Distribution service loaded for {year}")
            except FileNotFoundError as e:
                print(f"  Warning: KD service not available for {year} — {e}")

        except Exception as e:
            print(f"Warning: Failed to load model year {year} — {e}")

    print(f"Loaded {len(available_models)} model(s): {available_models}")

    # Load Multi-Output model (separate two-step XGBoost model)
    try:
        mo_dir = _resolve_model_dir("multi_output")
        if mo_dir and os.path.exists(os.path.join(mo_dir, "baseline_yield_model.joblib")):
            mo_service = MultiOutputService(mo_dir)
            print(f"Multi-Output model loaded ({len(mo_service.bin_cols)} BIN categories)")
        else:
            print("Warning: Multi-Output model not found, /predict-multi-output will be unavailable")
    except Exception as e:
        print(f"Warning: Multi-Output model failed to load — {e}")
        mo_service = None

    # Load KD ML Distribution model (KNN + XGBoost)
    try:
        kd_ml_dir = _resolve_model_dir("kd_distribution")
        if kd_ml_dir and os.path.exists(os.path.join(kd_ml_dir, "kd_model_bundle.joblib")):
            kd_ml_service = KDMLDistributionService(kd_ml_dir)
            print(f"KD ML Distribution model loaded ({len(kd_ml_service.kd_cols)} KD categories)")
        else:
            print("Warning: KD ML model not found, /predict-kd-ml will be unavailable")
    except Exception as e:
        print(f"Warning: KD ML model failed to load — {e}")
        kd_ml_service = None

    yield  # App runs here
    print("Shutting down...")


app = FastAPI(
    title="Yield Prediction API",
    description="Production ML inference service — supports multiple model years (2024, 2025, 2year) and multi-output prediction",
    version="2.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authenticated router — all routes except /health require Basic Auth
auth_router = APIRouter(dependencies=[Depends(verify_credentials)])


# ============== Yield Prediction ==============

@auth_router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    model_year: str = Query("2024", description="Model year: 2024, 2025, or 2year"),
):
    """
    Predict yield percentage and output board feet.

    Use `model_year` query param to select which model to use.
    Default: 2024.
    """
    start_time = time.perf_counter()

    ms = _get_model_or_raise(model_year, model_services, "Model service")
    pp = _get_model_or_raise(model_year, preprocessors, "Preprocessor")

    try:
        features = pp.encode(request)
        yield_pct = ms.predict(features)
        output_bf = request.total_input_bf * (yield_pct / 100)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            predicted_yield_pct=round(yield_pct, 2),
            predicted_output_bf=round(output_bf, 2),
            confidence_lower=round(max(0, yield_pct - CONFIDENCE_MARGIN), 2),
            confidence_upper=round(min(150, yield_pct + CONFIDENCE_MARGIN), 2),
            latency_ms=round(latency_ms, 2),
            model_version=model_year,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ============== Health & Info ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check — reports status of all loaded models."""
    return HealthResponse(
        status="healthy" if available_models else "unhealthy",
        model_loaded=len(available_models) > 0,
        model_version=", ".join(available_models) if available_models else "none",
    )


@auth_router.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Yield Prediction API",
        "version": "2.0.0",
        "available_models": available_models,
        "multi_output_available": mo_service is not None,
        "kd_ml_available": kd_ml_service is not None,
        "docs": "/docs",
        "health": "/health",
    }


@auth_router.get("/models", tags=["Info"])
async def list_models():
    """List all available model years and capabilities."""
    return {
        "available_models": available_models,
        "multi_output_available": mo_service is not None,
        "kd_models": list(kd_services.keys()),
    }


@auth_router.get("/valid-values/{field}", tags=["Info"])
async def get_valid_values(
    field: str,
    model_year: str = Query("2024", description="Model year"),
):
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
            detail=f"Unknown field: {field}. Valid fields: {list(field_map.keys())}",
        )

    pp = _get_model_or_raise(model_year, preprocessors, "Preprocessor")
    values = pp.get_valid_values(field_map[field])
    return {"field": field, "model_year": model_year, "valid_values": values, "count": len(values)}


# ============== KD Distribution Prediction ==============

@auth_router.post("/predict-kd", response_model=KDPredictionResponse, tags=["KD Prediction"])
async def predict_kd(
    request: KDPredictionRequest,
    model_year: str = Query("2024", description="Model year: 2024, 2025, or 2year"),
):
    """
    Predict KD material distribution with grade breakdown.

    Use `model_year` query param to select which model to use.
    """
    start_time = time.perf_counter()

    kd = _get_model_or_raise(model_year, kd_services, "KD Distribution service")

    try:
        result = kd.get_kd_prediction(
            input_material=request.input_material,
            plant=request.input_plant,
            input_bf=request.total_input_bf,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        latency_ms = (time.perf_counter() - start_time) * 1000

        return KDPredictionResponse(
            input_material=result["input_material"],
            input_plant=result["input_plant"],
            input_bf=result["input_bf"],
            summary=result["summary"],
            kd_outputs=result["kd_outputs"],
            latency_ms=round(latency_ms, 2),
            model_version=model_year,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KD prediction error: {str(e)}")


@auth_router.get("/kd-materials", tags=["KD Prediction"])
async def list_kd_materials(
    model_year: str = Query("2024", description="Model year"),
):
    """List available KS materials for KD prediction."""
    kd = _get_model_or_raise(model_year, kd_services, "KD Distribution service")
    materials = kd.get_available_materials()
    return {"model_year": model_year, "materials": materials, "count": len(materials)}


@auth_router.get("/kd-plants/{material}", tags=["KD Prediction"])
async def list_kd_plants(
    material: str,
    model_year: str = Query("2024", description="Model year"),
):
    """List available plants for a specific KS material."""
    kd = _get_model_or_raise(model_year, kd_services, "KD Distribution service")

    plants = kd.get_available_plants(material)
    if not plants:
        raise HTTPException(status_code=404, detail=f"Material not found: {material}")

    return {"model_year": model_year, "material": material, "plants": plants, "count": len(plants)}


# ============== Multi-Output Prediction ==============

@auth_router.post("/predict-multi-output", response_model=MultiOutputResponse, tags=["Multi-Output Prediction"])
async def predict_multi_output(request: MultiOutputRequest):
    """
    Predict output board distribution from 261 consumption tally records.

    Two-step XGBoost model:
    1. Predicts yield factor (what fraction of input becomes output)
    2. Predicts BIN distribution (how output splits across Grade/Length/Width)

    Supports batch processing (up to 10,000 records).
    Records are automatically grouped by MANUFACTURINGORDER.
    """
    start_time = time.perf_counter()

    if mo_service is None:
        raise HTTPException(
            status_code=503,
            detail="Multi-Output model not available. Ensure baseline_yield_model.joblib and training_template.joblib are in models/multi_output/",
        )

    try:
        records = [r.model_dump() for r in request.records]

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, mo_service.predict, records),
            timeout=120.0,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return MultiOutputResponse(
            num_orders=result["num_orders"],
            total_bfin=result["total_bfin"],
            avg_yield_factor=result["avg_yield_factor"],
            total_predicted_output=result["total_predicted_output"],
            total_boards=result["total_boards"],
            orders=result["orders"],
            latency_ms=round(latency_ms, 2),
            model_version="multi-output-v1",
            records_received=result.get("records_received"),
            records_processed=result.get("records_processed"),
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Prediction timed out after 120s for {len(request.records)} records",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-output prediction error: {str(e)}")


# ============== KD ML Distribution Prediction ==============

@auth_router.post("/predict-kd-ml", response_model=KDMLPredictionResponse, tags=["KD ML Prediction"])
async def predict_kd_ml(request: KDMLPredictionRequest):
    """
    Predict KD material distribution using ML model (KNN + XGBoost).

    Accepts bulk tally records. Records are **grouped by manufacturing_order**
    and all rows in the same order are processed together (matching how the
    Streamlit CSV upload works).

    Each order produces one prediction result with the full KD distribution.
    """
    start_time = time.perf_counter()

    if kd_ml_service is None:
        raise HTTPException(
            status_code=503,
            detail="KD ML model not available. Ensure kd_model_bundle.joblib, kd_training_template.joblib, and kd_material_history.joblib are in models/kd_distribution/",
        )

    try:
        items = [item.model_dump() for item in request.items]

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, kd_ml_service.predict, items),
            timeout=120.0,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return KDMLPredictionResponse(
            results=result["results"],
            num_orders=result["num_orders"],
            total_records=result["total_records"],
            successful=result["successful"],
            failed=result["failed"],
            errors=result["errors"],
            latency_ms=round(latency_ms, 2),
            model_version="kd-ml-v1",
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Prediction timed out after 120s for {len(request.items)} records",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KD ML prediction error: {str(e)}")


# Register all authenticated routes
app.include_router(auth_router)
