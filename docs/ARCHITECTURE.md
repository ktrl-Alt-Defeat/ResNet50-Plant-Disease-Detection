# System Architecture — ResNet-50 Crop Disease Detection

This document specifies the end-to-end system architecture, module relationships, execution flows, and component interactions of the ResNet-50 Crop Disease Detection repository.

---

## 🏛️ High-Level System Architecture

The repository is structured into two main operational subsystems:
1. **Offline ML Pipeline (`src/`)**: Data validation, dataset preparation, custom ResNet-50 model training, evaluation, and benchmarking.
2. **Online Serving Subsystem (`deployment/`)**: Production FastAPI REST API, singleton model engine, image verification, and response serialization.

```mermaid
flowchart TD
    subgraph Storage ["Dataset & Storage Layer"]
        DataDir["data/ (train, val, test)"]
        ClassJSON["results/class_to_idx.json"]
        Checkpoints["checkpoints/best_model.pt"]
        HFHub["Hugging Face Hub (kanish33/resnet50)"]
    end

    subgraph OfflinePipeline ["Offline Machine Learning Pipeline (src/)"]
        Validation["dataset_validation.py"]
        DatasetLoader["dataset.py (MappedImageFolder)"]
        ModelArch["model.py (ResNet50 & Bottleneck)"]
        Trainer["train.py (AdamW, Cosine, AMP, EarlyStopping)"]
        Evaluator["evaluate.py & metrics.py"]
        Benchmarker["benchmark.py"]
    end

    subgraph OnlineServing ["Online REST API Subsystem (deployment/)"]
        FastAPIApp["app.py (FastAPI Lifespan & Endpoints)"]
        ModelEngine["model.py (ModelEngine Singleton)"]
        Preprocessor["preprocessing.py (PIL Validation & Transforms)"]
        Schemas["schemas.py (Pydantic Models)"]
    end

    subgraph Clients ["Clients & Infrastructure"]
        HTTPClient["HTTP Client / Web App"]
        DockerContainer["Docker Container (Python 3.10-slim)"]
        RenderService["Render Web Service"]
    end

    DataDir --> Validation
    Validation --> ClassJSON
    DataDir --> DatasetLoader
    ClassJSON --> DatasetLoader
    DatasetLoader --> Trainer
    ModelArch --> Trainer
    Trainer --> Checkpoints
    Checkpoints --> Evaluator
    HFHub -.->|Fallback Download| ModelEngine
    Checkpoints --> ModelEngine
    ClassJSON --> ModelEngine
    HTTPClient -->|POST /predict| FastAPIApp
    FastAPIApp --> Preprocessor
    Preprocessor --> ModelEngine
    ModelEngine --> Schemas
    Schemas -->|JSON Response| HTTPClient
    DockerContainer --> FastAPIApp
    RenderService --> DockerContainer
```

---

## 🧩 Module Relationship Matrix

| Module / Component | Import Dependencies | Dependent Modules | Core Responsibility | Source File Reference |
| :--- | :--- | :--- | :--- | :--- |
| **`src/utils.py`** | `yaml`, `torch`, `numpy`, `random`, `logging`, `csv` | `dataset.py`, `dataset_validation.py`, `model.py`, `train.py`, `evaluate.py`, `deployment/model.py` | Config loading, random seed reproducibility, GPU/CPU device detection, checkpoint loading/saving, logging. | [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py:L13-L181) |
| **`src/dataset_validation.py`** | `PIL.Image`, `hashlib`, `statistics`, `src.utils` | `tests/test_milestone2.py` | Data integrity scan, file corruption check, intra-split duplicate detection, cross-split leakage scan, class alignment. | [`src/dataset_validation.py`](file:///d:/resnet%20crop%20detection/src/dataset_validation.py:L15-L517) |
| **`src/dataset.py`** | `torchvision`, `torch.utils.data`, `PIL`, `src.utils` | `train.py`, `evaluate.py`, `model.py`, `tests/test_milestone2.py` | `MappedImageFolder` subclass, dynamic class inference, ImageNet transforms, DataLoader factory, sample visualization. | [`src/dataset.py`](file:///d:/resnet%20crop%20detection/src/dataset.py:L19-L306) |
| **`src/model.py`** | `torch`, `torch.nn`, `src.utils` | `train.py`, `evaluate.py`, `deployment/model.py`, `tests/test_milestone3.py` | Custom `ResNet50` & `Bottleneck` PyTorch blocks, `build_model`, weight initialization, parameter count sanity check. | [`src/model.py`](file:///d:/resnet%20crop%20detection/src/model.py:L13-L375) |
| **`src/train.py`** | `torch.optim`, `torch.amp`, `src.utils`, `src.model`, `src.dataset` | `tests/test_milestone4.py` | Full training loop, AdamW optimizer, Cosine Annealing, AMP GradScaler, EarlyStopping, atomic checkpointing. | [`src/train.py`](file:///d:/resnet%20crop%20detection/src/train.py:L28-L667) |
| **`src/metrics.py`** | `scikit-learn`, `matplotlib`, `PIL`, `numpy` | `evaluate.py`, `tests/test_milestone5.py` | Metric computation (Top-1/5, Macro/Weighted F1, AUROC, AUPRC, 15-bin ECE), plot generation (CM, Reliability, ROC, PR). | [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py:L30-L373) |
| **`src/evaluate.py`** | `src.utils`, `src.model`, `src.dataset`, `src.metrics`, `src.benchmark` | `tests/test_milestone5.py` | Standalone test set evaluation orchestrator, master report generator (`test_evaluation_report.json`). | [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py:L34-L276) |
| **`src/benchmark.py`** | `torch`, `numpy`, `time` | `evaluate.py` | Warm-up & timed inference latency/throughput benchmarking routine. | [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py:L9-L100) |
| **`deployment/app.py`** | `fastapi`, `deployment.schemas`, `deployment.model` | Docker container, Render | FastAPI application instance, lifespan model loading hook, GET `/`, GET `/health`, POST `/predict` routes. | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L12-L146) |
| **`deployment/model.py`** | `torch`, `src.model`, `src.utils`, `deployment.preprocessing` | `deployment/app.py` | `ModelEngine` singleton class, Hugging Face checkpoint auto-downloader, batch inference executor. | [`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py:L12-L150) |
| **`deployment/preprocessing.py`** | `PIL.Image`, `torchvision.transforms`, `io.BytesIO` | `deployment/model.py` | Byte stream verification, PIL RGB conversion, image dimension validation, center crop inference transforms. | [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py:L12-L65) |
| **`deployment/schemas.py`** | `pydantic` | `deployment/app.py` | Data validation schemas for API metadata, health status, prediction output, and error responses. | [`deployment/schemas.py`](file:///d:/resnet%20crop%20detection/deployment/schemas.py:L5-L41) |

---

## 🔄 Execution Flows

### 1. Data Processing & Validation Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI as CLI / User
    participant ValEngine as src/dataset_validation.py
    participant Utils as src/utils.py
    participant DS as src/dataset.py
    participant Disk as Disk (results/)

    CLI->>Utils: load_config("configs/config.yaml")
    Utils-->>CLI: config dict
    CLI->>ValEngine: run_full_dataset_validation(config)
    ValEngine->>ValEngine: check_split_integrity("train", "val", "test")
    ValEngine->>ValEngine: validate_class_alignment()
    ValEngine->>ValEngine: detect_duplicates_and_leakage()
    ValEngine->>Disk: Save results/class_to_idx.json
    ValEngine->>Disk: Save results/dataset_class_distribution.csv
    ValEngine->>Disk: Save results/dataset_validation_report.json
    CLI->>DS: build_dataloaders(config)
    DS->>DS: MappedImageFolder(root, master_class_to_idx)
    DS-->>CLI: train_loader, val_loader, test_loader, class_names, class_to_idx
```

---

### 2. Model Training & Atomic Checkpointing Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    participant TrainCLI as python -m src.train
    participant Trainer as src/train.py
    participant Model as src/model.py (ResNet50)
    participant Opt as AdamW & CosineScheduler
    participant Scaler as AMP GradScaler
    participant Disk as Checkpoint Directory

    TrainCLI->>Trainer: run_real_training(config)
    Trainer->>Model: build_model(config)
    Trainer->>Opt: build_optimizer() & build_scheduler()
    Trainer->>Scaler: GradScaler(enabled=cuda)
    
    loop Epoch 1 to 50
        Trainer->>Model: train_one_epoch()
        Note over Model: Forward pass in torch.amp.autocast()
        Scaler->>Model: scale(loss).backward()
        Scaler->>Opt: step(optimizer) & update()
        Trainer->>Model: validate_one_epoch() (no_grad)
        Trainer->>Trainer: EarlyStopping.step(val_loss)
        Trainer->>Disk: save_atomic_checkpoint(last_model.pt)
        opt New Best Val Loss
            Trainer->>Disk: save_atomic_checkpoint(best_model.pt)
        end
    end
```

---

### 3. REST API Image Prediction Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    participant Client as HTTP Client
    participant App as deployment/app.py
    participant Prep as deployment/preprocessing.py
    participant Engine as deployment/model.py (ModelEngine)
    participant PyTorch as ResNet-50 Model

    Note over App, Engine: Startup: Lifespan loads model once into memory
    Client->>App: POST /predict (multipart file upload)
    App->>App: Validate file extension (.jpg, .jpeg, .png, .bmp, .webp)
    alt Invalid Extension
        App-->>Client: HTTP 415 Unsupported Media Type
    end
    App->>Prep: load_and_validate_image(file_bytes)
    Prep->>Prep: PIL Image.open() & img.verify() & check dims > 0
    alt Corrupt or Invalid Image Bytes
        Prep-->>App: raise ValueError
        App-->>Client: HTTP 400 Bad Request
    end
    Prep->>Prep: convert("RGB") -> get_inference_transforms()
    Prep-->>Engine: Tensor [1, 3, 224, 224]
    Engine->>PyTorch: forward pass (torch.inference_mode)
    PyTorch-->>Engine: Logits tensor [1, num_classes]
    Engine->>Engine: softmax() -> top-k probabilities (k=5)
    Engine-->>App: Result dict (predicted_class, confidence, top_5_predictions)
    App-->>Client: 200 OK Response (PredictionResponse Pydantic JSON)
```

---

## 🔒 Component Boundaries & Operational Policies

1. **Test Dataset Protection Policy**: The test dataset (`data/test`) is strictly isolated and **never** used during training, validation, early stopping, or checkpoint selection ([`src/train.py`](file:///d:/resnet%20crop%20detection/src/train.py:L500-L502), [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py:L236-L239)).
2. **Master Class Mapping Authority**: Class mapping relies on [`results/class_to_idx.json`](file:///d:/resnet%20crop%20detection/results/class_to_idx.json) created during dataset validation. Both training and deployment load this file as the authoritative class index mapping ([`src/dataset.py`](file:///d:/resnet%20crop%20detection/src/dataset.py:L61-L75), [`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py:L63-L67)).
3. **Model Weight Loading Authority**: At deployment startup, `ModelEngine` loads local weights from `checkpoints/best_model.pt`. If missing, it downloads the weights from Hugging Face Hub ([`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py:L29-L54)).
