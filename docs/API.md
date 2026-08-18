# FastAPI REST API Specification — ResNet-50 Crop Disease Detection

This document provides the complete API specification for the production FastAPI serving application implemented in [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py).

---

## 🌐 Service Overview

- **Application Title**: ResNet-50 Crop Disease Classification API
- **Framework**: FastAPI 0.100+ running on Uvicorn
- **Default Port**: `8080` (overridden via environment variable `PORT`)
- **Documentation URLs**: Interactive Swagger UI at `/docs`, ReDoc UI at `/redoc`
- **Lifespan Engine**: Pre-loads PyTorch `ResNet-50` model ONCE into memory during server startup using `asynccontextmanager` ([`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L12-L28)).

---

## 🔒 Security & Middleware Configuration

### 1. Authentication & Authorization
- **Status**: **Not Found**
- **Details**: No authentication middleware, API key validation, JWT bearer tokens, or user role access controls are implemented in [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py). All endpoints are publicly accessible.

### 2. CORS (Cross-Origin Resource Sharing)
Configured via `CORSMiddleware` in [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L40-L54):
- **Allowed Origins**: `http://localhost`, `http://localhost:3000`, `http://localhost:8000`, `http://127.0.0.1:3000`, `http://127.0.0.1:8000`
- **Allowed Credentials**: `True`
- **Allowed Methods**: `["*"]`
- **Allowed Headers**: `["*"]`

---

## 📡 REST API Endpoints

### 1. Root / API Info Endpoint

- **HTTP Method**: `GET`
- **Path**: `/`
- **Summary**: Get API Info
- **Response Model**: [`APIInfoResponse`](file:///d:/resnet%20crop%20detection/deployment/schemas.py:5-10)

#### Response Example (200 OK):
```json
{
  "name": "ResNet-50 Crop Disease Classification API",
  "model_name": "ResNet-50",
  "num_classes": 38,
  "status": "online"
}
```

---

### 2. Health Check Endpoint

- **HTTP Method**: `GET`
- **Path**: `/health`
- **Summary**: Health Check
- **Response Model**: [`HealthResponse`](file:///d:/resnet%20crop%20detection/deployment/schemas.py:13-18)

#### Response Example (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### 3. Image Classification / Inference Endpoint

- **HTTP Method**: `POST`
- **Path**: `/predict`
- **Summary**: Classify Crop Disease Image
- **Consumes**: `multipart/form-data`
- **Request Body**:
  - `file`: `UploadFile` (Multipart image file upload)
- **Supported File Extensions**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` ([`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:109))
- **Response Model**: [`PredictionResponse`](file:///d:/resnet%20crop%20detection/deployment/schemas.py:28-33)

#### Successful Response Example (200 OK):
```json
{
  "predicted_class": "Tomato___Target_Spot",
  "confidence": 0.9842,
  "confidence_percentage": 98.42,
  "top_5_predictions": [
    {
      "class": "Tomato___Target_Spot",
      "confidence": 0.9842
    },
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.0115
    },
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.0028
    },
    {
      "class": "Tomato___Leaf_Mold",
      "confidence": 0.0011
    },
    {
      "class": "Tomato___Bacterial_spot",
      "confidence": 0.0004
    }
  ]
}
```

---

## ⚠️ Input Validation & Exception Handling

Input validation is executed in a multi-stage process prior to running PyTorch inference ([`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:106-146), [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py:25-55)):

| Step | Validation Rule | Exception Code | Returned Detail Message | Source File |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Extension check against `{"jpg", "jpeg", "png", "bmp", "webp"}` | `415 Unsupported Media Type` | `Unsupported image file extension: '.{ext}'. Supported extensions: [...]` | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L106-L114) |
| 2 | Check file bytes are non-empty | `400 Bad Request` | `Uploaded file is empty.` | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L125-L129) |
| 3 | PIL image open and integrity check (`img.verify()`) | `400 Bad Request` | `Invalid or corrupted image format (...)` | [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py:L34-L54) |
| 4 | Non-zero dimension check (`width > 0` and `height > 0`) | `400 Bad Request` | `Invalid image dimensions: {w}x{h}` | [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py:L44-L47) |
| 5 | General inference exception catch | `500 Internal Server Error` | `Inference pipeline error: {str(e)}` | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L141-L146) |

---

## 📋 Data Schemas (Pydantic Models)

Defined in [`deployment/schemas.py`](file:///d:/resnet%20crop%20detection/deployment/schemas.py):

```python
class APIInfoResponse(BaseModel):
    name: str
    model_name: str
    num_classes: int
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class PredictionItem(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    confidence_percentage: float
    top_5_predictions: List[PredictionItem]

class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    status_code: int
```

---

## 💻 Request Code Examples

### 1. cURL Example
```bash
# Health Check
curl -X GET "http://localhost:8080/health"

# Predict Image
curl -X POST "http://localhost:8080/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/leaf_sample.jpg"
```

### 2. Python Requests Example
```python
import requests

# Health Check
health_res = requests.get("http://localhost:8080/health")
print("Health:", health_res.json())

# Predict Image
url = "http://localhost:8080/predict"
with open("leaf_sample.jpg", "rb") as img_file:
    files = {"file": ("leaf_sample.jpg", img_file, "image/jpeg")}
    response = requests.post(url, files=files)

print("Prediction Status Code:", response.status_code)
print("Prediction Result:", response.json())
```
