from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deployment.schemas import APIInfoResponse, HealthResponse, PredictionResponse, ErrorResponse
from deployment.model import model_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Loads the ResNet-50 PyTorch model ONCE into memory during application startup.
    """
    print("\n==================================================")
    print("[FastAPI Startup] Initializing ResNet-50 Inference Engine...")
    print("==================================================")
    try:
        model_engine.load_model()
    except Exception as e:
        print(f"[FastAPI Startup ERROR] Failed to load model: {e}")
        raise RuntimeError(f"Application failed to initialize model: {e}")
    yield
    print("[FastAPI Shutdown] Cleaning up resources.")


app = FastAPI(
    title="ResNet-50 Crop Disease Classification API",
    description="Production-ready REST API for plant/crop leaf disease detection using custom ResNet-50.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for local frontend integration
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=APIInfoResponse,
    tags=["Root"],
    summary="Get API Info"
)
async def get_api_info() -> APIInfoResponse:
    """Return basic API information, model architecture, and total class count."""
    return APIInfoResponse(
        name="ResNet-50 Crop Disease Classification API",
        model_name="ResNet-50",
        num_classes=model_engine.num_classes if model_engine.is_loaded else 38,
        status="online"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check"
)
async def health_check() -> HealthResponse:
    """Return system health status, model loading state, and active PyTorch device."""
    return HealthResponse(
        status="healthy" if model_engine.is_loaded else "unhealthy",
        model_loaded=model_engine.is_loaded,
        device=str(model_engine.device)
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request — Invalid or corrupt image file"},
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        500: {"model": ErrorResponse, "description": "Internal Server Error during inference"}
    },
    tags=["Inference"],
    summary="Classify Crop Disease Image"
)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Classify an uploaded crop/leaf image.
    
    - **file**: Multipart image upload (.jpg, .jpeg, .png, .webp, .bmp)
    - **Returns**: Top-1 predicted class, confidence, confidence percentage, and top-5 predictions.
    """
    # 1. Content-Type and Filename Validation
    if file.filename:
        ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        valid_exts = {"jpg", "jpeg", "png", "bmp", "webp"}
        if ext and ext not in valid_exts:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported image file extension: '.{ext}'. Supported extensions: {sorted(list(valid_exts))}"
            )

    # 2. Read File Bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 3. Perform Inference
    try:
        result = model_engine.predict(file_bytes, top_k=5)
        return PredictionResponse(**result)
    except ValueError as ve:
        # Invalid image bytes / corrupt file
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Avoid exposing stack trace
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {str(e)}"
        )
