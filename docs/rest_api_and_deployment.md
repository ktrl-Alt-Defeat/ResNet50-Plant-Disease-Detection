# REST API & Cloud Deployment Documentation

This document provides a reverse engineered specification of the production FastAPI web service, singleton inference engine, image validation pipeline, Pydantic schemas, Docker containerization, and Render deployment pipeline implemented in `deployment/` and root deployment manifests.

---

## 🌐 FastAPI Application Architecture (`deployment/app.py`)

- **Framework**: FastAPI 0.100+ running on Uvicorn web server.
- **Lifespan Manager**: Uses `@asynccontextmanager` lifespan event to initialize PyTorch weights ONCE on startup into CPU/GPU memory:
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      model_engine.load_model()
      yield
  ```
- **CORS Configuration**: Configured via `CORSMiddleware` supporting origins (`localhost`, `3000`, `8000`).

---

## 🔌 API Endpoints & Schemas (`deployment/schemas.py`)

### 1. `GET /` — API Information
- **Response Schema**: `APIInfoResponse`
- **Output**: Returns API name, model name (`ResNet-50`), total class count, and status (`online`).

### 2. `GET /health` — Health Probe
- **Response Schema**: `HealthResponse`
- **Output**: Returns status (`healthy` or `unhealthy`), model loaded state (`True`/`False`), and device string (`cpu` or `cuda:0`).

### 3. `POST /predict` — Image Classification Endpoint
- **Request Format**: `multipart/form-data` with `file` upload.
- **Accepted File Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- **Response Schema**: `PredictionResponse`
  ```json
  {
    "predicted_class": "Tomato___Bacterial_spot",
    "confidence": 0.9842,
    "confidence_percentage": 98.42,
    "top_5_predictions": [
      { "class": "Tomato___Bacterial_spot", "confidence": 0.9842 },
      { "class": "Tomato___Early_blight", "confidence": 0.0112 }
    ]
  }
  ```

---

## 🧠 Model Inference Engine & Hugging Face Integration (`deployment/model.py`)

### Singleton Pattern (`ModelEngine`)
- **Instantiation**: Single global object `model_engine = ModelEngine()`.
- **Model Checkpoint Auto-Download**:
  If `checkpoints/best_model.pt` does not exist locally or inside the Docker container, `_download_checkpoint_from_hf()` automatically downloads the model weights from Hugging Face Hub:
  `https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt`

### Image Preprocessing & Inference Pipeline (`deployment/preprocessing.py`)
1. Raw HTTP file bytes $\to$ `io.BytesIO`.
2. PIL verification via `Image.open().verify()` and non-zero dimension check.
3. Convert to 3-channel RGB (`img.convert("RGB")`).
4. Apply evaluation transformations (`Resize(256)` $\to$ `CenterCrop(224)` $\to$ `ToTensor()` $\to$ `Normalize()`).
5. Forward pass with `@torch.inference_mode()`.
6. Apply Softmax $\to$ Extract Top-K probabilities via `torch.topk()`.

---

## 🐳 Docker & Render Deployment (`Dockerfile`, `render.yaml`)

### Docker Container Specification ([`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile))
- **Base Image**: `python:3.10-slim`.
- **PyTorch Optimization**: Installs CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) to keep container image lightweight (~700MB) and prevent OOM errors on cloud free/starter tiers.
- **Model Pre-fetching**: Pre-downloads `best_model.pt` from Hugging Face during image build step (`RUN python -c "import urllib.request..."`).
- **Dynamic Port Binding**: Uvicorn executes via shell expansion `CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-10000}"]` to bind dynamically to Render's `$PORT`.

### Render Blueprint ([`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml))
- Configured as a Docker Web Service on Render with `/health` check routing.
