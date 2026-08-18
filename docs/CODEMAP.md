# Codebase Navigation Map & Technical Debt Audit

This document provides a catalog of every important directory, file, class, function, line range, and responsibility in the repository, along with an explicit technical debt audit.

---

## 🗺️ Codebase File Catalog

### 1. Offline ML Subsystem (`src/`)

| File Path | Primary Responsibilities | Classes & Functions Implemented | Source File Reference |
| :--- | :--- | :--- | :--- |
| [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py) | Configuration loading, random seed initialization, GPU device selection, logging, and checkpoint file saving/loading. | `load_config()`, `save_config()`, `get_device()`, `set_seed()`, `save_checkpoint()`, `load_checkpoint()`, `ExperimentLogger` | [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py:L13-L181) |
| [`src/dataset.py`](file:///d:/resnet%20crop%20detection/src/dataset.py) | Custom PyTorch `MappedImageFolder` dataset class, dynamic class folder scanner, transform builder, DataLoader generator, batch verification, and PIL grid visualizer. | `MappedImageFolder`, `infer_num_classes()`, `get_class_names()`, `get_class_mapping()`, `get_transforms()`, `create_datasets()`, `build_dataloaders()`, `verify_dataloader_batches()`, `visualize_dataset_samples()` | [`src/dataset.py`](file:///d:/resnet%20crop%20detection/src/dataset.py:L19-L306) |
| [`src/dataset_validation.py`](file:///d:/resnet%20crop%20detection/src/dataset_validation.py) | Dataset integrity scanner, file corruption check, intra-split exact duplicate scanner (SHA-256), perceptual near-duplicate scanner (dhash), cross-split data leakage detector, and class distribution calculator. | `compute_sha256()`, `compute_dhash()`, `validate_single_image()`, `check_split_integrity()`, `detect_duplicates_and_leakage()`, `validate_class_alignment()`, `calculate_distribution_stats()`, `export_class_distribution_csv()`, `infer_and_save_class_mapping()`, `compute_image_size_stats()`, `run_full_dataset_validation()` | [`src/dataset_validation.py`](file:///d:/resnet%20crop%20detection/src/dataset_validation.py:L15-L517) |
| [`src/model.py`](file:///d:/resnet%20crop%20detection/src/model.py) | Custom ResNet-50 architecture definition, `Bottleneck` residual block, weight initialization, feature extractor, classifier head, parameter counter, model builder, and architecture summary generator. | `Bottleneck`, `ResNet50`, `get_parameter_count()`, `sanity_check_parameter_count()`, `resolve_num_classes()`, `build_model()`, `generate_architecture_summary()`, `verify_model_pipeline()` | [`src/model.py`](file:///d:/resnet%20crop%20detection/src/model.py:L13-L536) |
| [`src/train.py`](file:///d:/resnet%20crop%20detection/src/train.py) | Full training orchestrator, loss function builder, AdamW optimizer builder, Cosine Annealing scheduler builder, AMP GradScaler, Early Stopping handler, atomic checkpoint saver (`os.replace`), and synthetic step verifier. | `validate_training_config()`, `check_dataset_counts()`, `build_loss()`, `build_optimizer()`, `build_scheduler()`, `save_atomic_checkpoint()`, `load_training_checkpoint()`, `EarlyStopping`, `train_one_epoch()`, `validate_one_epoch()`, `log_metrics_to_csv()`, `run_synthetic_step_verification()`, `visualize_training_curves()`, `run_real_training()` | [`src/train.py`](file:///d:/resnet%20crop%20detection/src/train.py:L28-L667) |
| [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py) | Metric calculation functions (Top-1/5, Macro/Weighted F1, One-vs-Rest AUROC/AUPRC, 15-bin ECE), confusion matrix export, calibration reliability diagram generator, ROC & PR curve plot generators. | `calculate_accuracy()`, `calculate_top_k_accuracy()`, `calculate_macro_metrics()`, `calculate_weighted_metrics()`, `calculate_per_class_metrics()`, `calculate_auroc_auprc()`, `calculate_ece()`, `plot_calibration_curve()`, `plot_roc_curves()`, `plot_pr_curves()`, `generate_confusion_matrix_artifacts()` | [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py:L30-L373) |
| [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py) | Milestone 5 test evaluation orchestrator, test dataset inference runner, parameter immutability verifier, per-class CSV report exporter, and master JSON report generator. | `evaluate_model_on_test_set()` | [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py:L34-L276) |
| [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py) | Standardized inference efficiency benchmarker measuring warmup, latency percentiles (avg, p50, p95), throughput, and peak VRAM allocation. | `run_standard_benchmark()` | [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py:L9-L100) |

---

### 2. Online Serving Subsystem (`deployment/`)

| File Path | Primary Responsibilities | Classes & Functions Implemented | Source File Reference |
| :--- | :--- | :--- | :--- |
| [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py) | FastAPI web application, lifespan context manager, CORS middleware setup, GET `/`, GET `/health`, and POST `/predict` REST routes. | `lifespan()`, `get_api_info()`, `health_check()`, `predict_image()` | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L12-L146) |
| [`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py) | Singleton `ModelEngine` class, local checkpoint loader, Hugging Face checkpoint downloader, and batch prediction runner. | `ModelEngine`, `_download_checkpoint_from_hf()`, `load_model()`, `predict()` | [`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py:L12-L150) |
| [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py) | Evaluation image transform constructor (`Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize()`), raw byte loading, PIL verification, dimension validation, and RGB conversion. | `get_inference_transforms()`, `load_and_validate_image()`, `preprocess_image_bytes()` | [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py:L12-L65) |
| [`deployment/schemas.py`](file:///d:/resnet%20crop%20detection/deployment/schemas.py) | Pydantic response models for API metadata, health status, individual class predictions, top-5 prediction lists, and error responses. | `APIInfoResponse`, `HealthResponse`, `PredictionItem`, `PredictionResponse`, `ErrorResponse` | [`deployment/schemas.py`](file:///d:/resnet%20crop%20detection/deployment/schemas.py:L5-L41) |

---

### 3. Configuration & Infrastructure

| File Path | Primary Responsibilities | Source File Reference |
| :--- | :--- | :--- |
| [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml) | YAML configuration file defining seed, data paths, model settings, optimizer/scheduler hyper-parameters, metrics, and output paths. | [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml:L1-L101) |
| [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile) | Production Docker image build file using `python:3.10-slim-bookworm`, CPU PyTorch, pre-downloaded weights, and healthcheck. | [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile:L1-L51) |
| [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml) | Render Infrastructure-as-Code Blueprint configuring docker service, region, health check path, and environment variables. | [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml:L1-L17) |
| [`requirements.txt`](file:///d:/resnet%20crop%20detection/requirements.txt) | Core framework dependencies required for training, data processing, metrics, and evaluation. | [`requirements.txt`](file:///d:/resnet%20crop%20detection/requirements.txt:L1-L17) |
| [`deployment/requirements.txt`](file:///d:/resnet%20crop%20detection/deployment/requirements.txt) | Lightweight dependencies required strictly for running the FastAPI web server. | [`deployment/requirements.txt`](file:///d:/resnet%20crop%20detection/deployment/requirements.txt:L1-L10) |

---

## 🛠️ Comprehensive Technical Debt & Quality Audit

### 1. TODO & FIXME Comments

| File Path | Line | Code Comment / Text | Category | Impact & Technical Debt Assessment |
| :--- | :--- | :--- | :--- | :--- |
| [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py#L117) | L117 | `print("[Benchmark] Note: Model implementation (ResNet-50) will be attached in the next milestone.")` | Legacy Comment | Leftover milestone development note in standalone CLI print block. Does not affect import usage in [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py). |
| [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml#L8) | L8 | `# Change to starter if extra RAM/CPU is desired` | Deployment Note | Configuration comment reminding operator of RAM scaling options on Render platform. |

---

### 2. Standalone Execution & Import Inconsistencies

- **Module Import Error in `src/benchmark.py`**:
  - **Issue**: Line 105 in [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py#L105) contains `from utils import load_config, get_device` instead of `from src.utils import load_config, get_device`.
  - **Impact**: Running `python src/benchmark.py` directly from project root fails with `ModuleNotFoundError: No module named 'utils'`.
  - **Workaround**: When imported from [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py:L18), `run_standard_benchmark` works cleanly without executing the `if __name__ == "__main__":` block.
- **Inconsistent Fallback Class Count Default**:
  - **Issue**: [`deployment/app.py:L68`](file:///d:/resnet%20crop%20detection/deployment/app.py#L68) returns fallback `38` if model is not loaded, whereas [`src/model.py:L269`](file:///d:/resnet%20crop%20detection/src/model.py#L269) returns fallback `124`.
  - **Impact**: Aesthetic discrepancy in root endpoint metadata before model engine initialization.

---

### 3. Missing Security Features

- **Missing Authentication & Authorization**:
  - **Issue**: No API key headers, JWT tokens, or HTTP basic auth middleware exist in [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py).
  - **Impact**: Anyone with network access can send requests to `/predict`.
- **Missing Request Rate Limiting**:
  - **Issue**: No rate limiting middleware (e.g. `slowapi`) is attached to FastAPI.
  - **Impact**: Vulnerable to Denial-of-Service (DoS) via rapid image upload requests.

---

### 4. Root Directory Artifacts

- **`results-1.zip` in Root**:
  - **Issue**: File [`results-1.zip`](file:///d:/resnet%20crop%20detection/results-1.zip) (1.48 MB) exists in the repository root directory.
  - **Impact**: Archived results zip file retained in version control alongside unzipped [`results/`](file:///d:/resnet%20crop%20detection/results) directory.
