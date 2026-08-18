# System Architecture Overview

This document provides a high-level reverse engineered breakdown of the **ResNet-50 Plant Disease Detection System**, detailing the end-to-end data lifecycle, core component couplings, and design principles.

---

## 🏛️ End-to-End System Lifecycle

```
[ Raw Leaf Images (JPEG/PNG) ]
             │
             ▼
[ Data & Validation Pipeline ] ──► (Detect Corrupt Images & Cross-Split Leakage)
             │
             ▼
[ Data Loaders (Train / Val / Test) ] ──► (PyTorch DataLoader with Custom Augmentations)
             │
             ▼
[ Custom ResNet-50 PyTorch Model ] ──► (23.5M Parameters, Bottleneck Blocks, Zero-Init BN)
             │
   ┌─────────┴─────────┐
   ▼                   ▼
[ Training Engine ]   [ Evaluation & Benchmark Engine ]
   │                   │
   ▼                   ▼
(AMP, AdamW, Cosine)  (Macro F1, ECE Calibration, Latency Benchmark)
   │
   ▼
[ Checkpoint Storage ] ──► (checkpoints/best_model.pt & Hugging Face Hub)
                                        │
                                        ▼
                             [ REST API & Render Cloud ] ──► (FastAPI, Uvicorn, Docker)
```

---

## 🧩 Core Subsystem Modules

### 1. Data Subsystem (`src/dataset.py`, `src/dataset_validation.py`)
- Responsible for directory scanning, image integrity checking, MD5/SHA-256 duplicate detection, class mapping generation, and PyTorch dataset/dataloader construction.
- Class mapping is saved as an authoritative mapping in `results/class_to_idx.json`.

### 2. Neural Network Backbone (`src/model.py`)
- Custom implementation of ResNet-50 architecture without reliance on pretrained torchvision backbones per project specifications.
- Consists of a 7x7 Conv Stem, MaxPool, 4 Bottleneck Stages (3, 4, 6, 3 blocks), Adaptive Global Average Pooling, and a dynamic Linear Classifier Head.

### 3. Training & Optimization Engine (`src/train.py`, `src/utils.py`)
- Manages complete model training loop featuring Automatic Mixed Precision (`torch.cuda.amp`), `AdamW` optimizer, `CosineAnnealingLR` scheduler, `CrossEntropyLoss`, gradient clipping, best-checkpoint saving, and early stopping.

### 4. Metric & Benchmarking Engine (`src/evaluate.py`, `src/metrics.py`, `src/benchmark.py`)
- Evaluates test predictions to generate confusion matrices, macro/weighted precision/recall/F1, AUROC curves, Expected Calibration Error (ECE) plots, and inference latency/throughput benchmarks.

### 5. Serving & Cloud Deployment Engine (`deployment/`, `Dockerfile`, `render.yaml`)
- A production FastAPI service featuring a singleton `ModelEngine` pattern.
- Incorporates automatic Hugging Face model checkpoint download (`https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt`) when deployed inside containerized cloud environments like Render.

---

## 🛠️ Key Design Patterns

- **Decoupled YAML Configuration**: Centralized settings in [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml) govern image dimensions, hyper-parameters, data paths, and metrics.
- **Dynamic Class Resolution**: Class counts and label indices are dynamically resolved from `results/class_to_idx.json` or checkpoint metadata.
- **Singleton Inference Engine**: `ModelEngine` in `deployment/model.py` loads weights once during application startup lifespan and reuses GPU/CPU tensors across API requests.
