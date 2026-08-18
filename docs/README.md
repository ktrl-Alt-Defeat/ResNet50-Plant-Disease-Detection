# ResNet-50 Crop Disease Detection System — Technical Documentation Index

Welcome to the comprehensive, reverse-engineered technical documentation for the **ResNet-50 Crop & Leaf Disease Detection System**. Every statement, configuration option, API route, and architectural workflow in this documentation suite is derived directly from source code inspection (`src/`, `deployment/`, `configs/`, `tests/`, `Dockerfile`, `render.yaml`).

---

## 🗺️ Documentation Sitemap

| Document | Focus & Description | Primary Source Files |
| :--- | :--- | :--- |
| **[ARCHITECTURE.md](file:///d:/resnet%20crop%20detection/docs/ARCHITECTURE.md)** | End-to-end system architecture, module relationships, execution flows, and sequence diagrams. | [`src/train.py`](file:///d:/resnet%20crop%20detection/src/train.py), [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py), [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py) |
| **[API.md](file:///d:/resnet%20crop%20detection/docs/API.md)** | FastAPI endpoints, Pydantic schemas, image validation rules, CORS, error responses, and authentication status. | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py), [`deployment/schemas.py`](file:///d:/resnet%20crop%20detection/deployment/schemas.py), [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py) |
| **[MODEL.md](file:///d:/resnet%20crop%20detection/docs/MODEL.md)** | Custom ResNet-50 PyTorch architecture, Bottleneck blocks, transforms, parameter counts, Hugging Face download, and singleton engine. | [`src/model.py`](file:///d:/resnet%20crop%20detection/src/model.py), [`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py), [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py) |
| **[METRICS.md](file:///d:/resnet%20crop%20detection/docs/METRICS.md)** | Top-1/Top-5 accuracy, Macro/Weighted F1, AUROC/AUPRC, 15-bin ECE calibration, evaluation pipeline, and visual artifact outputs. | [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py), [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py), [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py) |
| **[CONFIG.md](file:///d:/resnet%20crop%20detection/docs/CONFIG.md)** | YAML configuration schema, environment variables, random seed reproducibility, device selection, and feature flags. | [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml), [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py), [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile), [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml) |
| **[DEPLOYMENT.md](file:///d:/resnet%20crop%20detection/docs/DEPLOYMENT.md)** | Docker containerization details, Render Blueprint configuration, startup commands, and dependencies. | [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile), [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml), [`deployment/requirements.txt`](file:///d:/resnet%20crop%20detection/deployment/requirements.txt) |
| **[CODEMAP.md](file:///d:/resnet%20crop%20detection/docs/CODEMAP.md)** | Complete codebase map, file/class/function catalog, technical debt, TODOs, dead code, and undocumented components. | All repository files in [`src/`](file:///d:/resnet%20crop%20detection/src), [`deployment/`](file:///d:/resnet%20crop%20detection/deployment), [`configs/`](file:///d:/resnet%20crop%20detection/configs), [`tests/`](file:///d:/resnet%20crop%20detection/tests) |

---

## ⚡ Technology Stack Summary

| Layer | Component / Tool | Version / Details | Source Reference |
| :--- | :--- | :--- | :--- |
| **Deep Learning Framework** | PyTorch & Torchvision | `torch>=2.0.0`, `torchvision>=0.15.0` | [`requirements.txt`](file:///d:/resnet%20crop%20detection/requirements.txt:L1-L3), [`deployment/requirements.txt`](file:///d:/resnet%20crop%20detection/deployment/requirements.txt:L5-L6) |
| **Model Backbone** | Custom ResNet-50 | Bottleneck residual blocks (3, 4, 6, 3), zero-init BN | [`src/model.py`](file:///d:/resnet%20crop%20detection/src/model.py:L82-L132) |
| **REST API Server** | FastAPI & Uvicorn | `fastapi>=0.100.0`, `uvicorn[standard]>=0.22.0` | [`deployment/requirements.txt`](file:///d:/resnet%20crop%20detection/deployment/requirements.txt:L1-L2), [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py:L30-L37) |
| **Image Processing** | Pillow (PIL) | `pillow>=9.5.0` | [`requirements.txt`](file:///d:/resnet%20crop%20detection/requirements.txt:L7), [`deployment/preprocessing.py`](file:///d:/resnet%20crop%20detection/deployment/preprocessing.py:L25-L55) |
| **Numerical & Metrics** | NumPy & Scikit-Learn | `numpy>=1.24.0`, `scikit-learn>=1.0.0` | [`requirements.txt`](file:///d:/resnet%20crop%20detection/requirements.txt:L6-L13) |
| **Configuration** | PyYAML | `pyyaml>=6.0` | [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml:L1-L101), [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py:L13-L22) |
| **Model Storage** | Hugging Face Hub | `https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt` | [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile:L9), [`deployment/model.py`](file:///d:/resnet%20crop%20detection/deployment/model.py:L37) |
| **Containerization & Hosting** | Docker & Cloud Platforms | Docker (Python 3.10-slim), Railway (`railway.json`), Render Blueprint (`render.yaml`) | [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile), [`railway.json`](file:///d:/resnet%20crop%20detection/railway.json), [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml) |

---

## 📁 Repository Structure Overview

```
.
├── Dockerfile                  # Container build instructions for FastAPI serving
├── render.yaml                 # Render Infrastructure-as-Code Blueprint
├── requirements.txt            # Core framework dependencies for training/eval
├── configs/
│   └── config.yaml             # Master configuration file (data, model, train, metrics)
├── deployment/
│   ├── app.py                  # FastAPI application & REST endpoint routes
│   ├── model.py                # ModelEngine singleton inference engine
│   ├── preprocessing.py        # Image byte loading, verification & transform
│   ├── schemas.py              # Pydantic request/response schemas
│   └── requirements.txt        # Serving-only lightweight dependencies
├── docs/                       # Reverse engineered technical documentation
├── src/
│   ├── benchmark.py            # Latency & throughput benchmarking utility
│   ├── dataset.py              # PyTorch MappedImageFolder & DataLoader builder
│   ├── dataset_validation.py   # Dataset integrity, corruption & leakage scanner
│   ├── evaluate.py             # Milestone 5 test evaluation orchestrator
│   ├── metrics.py              # Metric calculation functions & plot generators
│   ├── model.py                # ResNet50 PyTorch architecture & builder
│   ├── train.py                # Training loop, AMP, early stopping & checkpointing
│   └── utils.py                # Config loader, device detection, seed & logging
└── tests/                      # Automated unit tests (Milestones 2 through 5)
    ├── test_milestone2.py      # Dataset & DataLoader verification tests
    ├── test_milestone3.py      # ResNet-50 architecture verification tests
    ├── test_milestone4.py      # Training pipeline & checkpointing tests
    └── test_milestone5.py      # Evaluation, metrics & inference tests
```
