# Configuration & Runtime Settings — ResNet-50 Crop Disease Detection

This document provides technical documentation for configuration files, environment variables, device selection rules, random seed reproducibility, and feature flags implemented in [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml), [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py), [`Dockerfile`](file:///d:/resnet%20crop%20detection/Dockerfile), and [`render.yaml`](file:///d:/resnet%20crop%20detection/render.yaml).

---

## 🌍 Environment Variables

Environment variables are set in [`Dockerfile:L5-L9`](file:///d:/resnet%20crop%20detection/Dockerfile#L5-L9), [`render.yaml:L12-L16`](file:///d:/resnet%20crop%20detection/render.yaml#L12-L16), and read dynamically by application scripts:

| Variable Name | Default Value | Target Subsystem | Description & Usage | Source Reference |
| :--- | :--- | :--- | :--- | :--- |
| `PORT` | `8080` (Docker) / `10000` (Render) | Deployment Server | Network port for Uvicorn web server execution. | [`Dockerfile:L8`](file:///d:/resnet%20crop%20detection/Dockerfile#L8), [`render.yaml:L14`](file:///d:/resnet%20crop%20detection/render.yaml#L14) |
| `HF_MODEL_URL` | `https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt` | `ModelEngine` | Fallback Hugging Face URL to download pre-trained PyTorch weights when `checkpoints/best_model.pt` is missing locally. | [`Dockerfile:L9`](file:///d:/resnet%20crop%20detection/Dockerfile#L9), [`deployment/model.py:L35-L38`](file:///d:/resnet%20crop%20detection/deployment/model.py#L35-L38) |
| `PYTHONUNBUFFERED` | `1` | Python Runtime | Disables Python stdout/stderr stream buffering for immediate real-time container log output. | [`Dockerfile:L5`](file:///d:/resnet%20crop%20detection/Dockerfile#L5) |
| `PYTHONDONTWRITEBYTECODE` | `1` | Python Runtime | Prevents Python from writing `.pyc` compiled bytecode files onto container filesystem. | [`Dockerfile:L6`](file:///d:/resnet%20crop%20detection/Dockerfile#L6) |
| `PYTHONPATH` | `/app` | Python Import Engine | Ensures `/app` root directory is included in module resolution path within Docker container. | [`Dockerfile:L7`](file:///d:/resnet%20crop%20detection/Dockerfile#L7) |
| `CORS_ORIGINS` | `*` | REST API Server | Comma-separated list of allowed origins or `*` for wildcard access (Railway, Vercel). | [`deployment/app.py:L41`](file:///d:/resnet%20crop%20detection/deployment/app.py#L41) |
| `PYTHONHASHSEED` | String value of `seed` (e.g. `"42"`) | Reproducibility | Enforces deterministic Python string hashing when `deterministic: True`. | [`src/utils.py:L89`](file:///d:/resnet%20crop%20detection/src/utils.py#L89) |

---

## ⚙️ Configuration File Schema (`configs/config.yaml`)

Main settings are stored in [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml):

| Top-Level Section | Key | Type | Default Value | Description & Codebase Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Global** | `seed` | Integer | `42` | Global random seed for `random`, `np.random`, `torch`, and CUDA ([`src/utils.py:L72`](file:///d:/resnet%20crop%20detection/src/utils.py#L72)). |
| **Global** | `deterministic` | Boolean | `true` | If `true`, enables `cudnn.deterministic=True` and `cudnn.benchmark=False`. |
| **Global** | `device` | String | `"auto"` | Hardware device preference (`"auto"`, `"cuda"`, or `"cpu"`) ([`src/utils.py:L33`](file:///d:/resnet%20crop%20detection/src/utils.py#L33)). |
| **`data`** | `data_dir` | String | `"data"` | Root data directory path. |
| | `train_dir` | String | `"data/train"` | Training split folder path. |
| | `val_dir` | String | `"data/val"` | Validation split folder path. |
| | `test_dir` | String | `"data/test"` | Test split folder path. |
| | `num_classes` | Null / Int | `null` | Target output class count (`null` triggers dynamic inference from train directory). |
| | `img_size` | Integer | `224` | Final crop dimension for network input (224x224). |
| | `resize_size` | Integer | `256` | Intermediate image resize dimension prior to cropping. |
| | `batch_size` | Integer | `32` | DataLoader mini-batch size. |
| | `num_workers` | Integer | `0` | PyTorch DataLoader worker subprocesses (`0` for Windows compatibility). |
| | `pin_memory` | Boolean | `true` | Enables page-locked memory transfers to GPU (`false` automatically on CPU). |
| | `drop_last` | Boolean | `false` | If `true`, drops un-filled final batch in DataLoader. |
| **`data.augmentation`**| `random_crop_scale`| List[float]| `[0.8, 1.0]` | Random crop scale bounds for training dataset augmentation. |
| | `horizontal_flip_prob`| Float | `0.5` | Random horizontal flip probability. |
| | `vertical_flip_prob` | Float | `0.2` | Random vertical flip probability. |
| | `rotation_degrees` | Integer | `15` | Random rotation range in degrees ($\pm 15^\circ$). |
| | `color_jitter` | Float | `0.2` | Brightness, contrast, saturation jitter factor. |
| **`data.normalization`**| `mean` | List[float]| `[0.485, 0.456, 0.406]` | ImageNet channel RGB mean values. |
| | `std` | List[float]| `[0.229, 0.224, 0.225]` | ImageNet channel RGB standard deviation values. |
| **`validation`** | `check_corrupted` | Boolean | `true` | Enables PIL corruption scanner during dataset validation. |
| | `detect_leakage` | Boolean | `true` | Enables SHA-256 duplicate cross-split data leakage scanner. |
| | `supported_extensions`| List[str] | `[".jpg", ".jpeg", ".png", ".bmp", ".webp"]` | Valid image file extension filter list. |
| **`model`** | `name` | String | `"resnet50"` | Backbone architecture identifier. |
| | `num_classes` | Null / Int | `null` | Resolved dynamically from `results/class_to_idx.json`. |
| | `pretrained` | Boolean | `false` | Un-pretrained baseline per M3 requirements. |
| | `dropout` | Float | `0.0` | Classification head dropout rate (`0.0` disables dropout). |
| | `zero_init_residual`| Boolean | `true` | Enables zero-initialization of final BN weight in bottleneck blocks. |
| **`training`** | `epochs` | Integer | `50` | Maximum training epoch count. |
| | `batch_size` | Integer | `32` | Training mini-batch size. |
| | `seed` | Integer | `42` | Training seed override. |
| **`optimizer`** | `name` | String | `"adamw"` | Optimizer algorithm (`"adamw"`, `"adam"`, or `"sgd"`). |
| | `learning_rate` | Float | `0.0003` | Initial learning rate. |
| | `weight_decay` | Float | `0.0001` | Weight decay coefficient (L2 regularization). |
| **`scheduler`** | `name` | String | `"cosine"` | Learning rate scheduler algorithm (`"cosine"` or `"step"`). |
| | `min_lr` | Float | `0.000001` | Minimum learning rate lower bound for cosine annealing. |
| **`loss`** | `name` | String | `"cross_entropy"` | Loss function identifier (`nn.CrossEntropyLoss`). |
| | `label_smoothing`| Float | `0.0` | Cross entropy label smoothing factor (`0.0` disables smoothing). |
| **`mixed_precision`** | `enabled` | Boolean | `true` | Enables Automatic Mixed Precision (AMP) `torch.amp.GradScaler` on CUDA. |
| **`gradient`** | `max_norm` | Float | `1.0` | Gradient clipping threshold norm (`torch.nn.utils.clip_grad_norm_`). |
| **`early_stopping`** | `enabled` | Boolean | `true` | Enables validation loss early stopping monitoring. |
| | `patience` | Integer | `10` | Number of epochs without improvement before stopping. |
| | `monitor` | String | `"val_loss"` | Metric monitored for early stopping. |
| | `mode` | String | `"min"` | Optimization target direction (`"min"` for loss). |
| **`checkpoint`** | `save_best` | Boolean | `true` | Automatically saves `checkpoints/best_model.pt`. |
| | `save_last` | Boolean | `true` | Automatically saves `checkpoints/last_model.pt`. |
| **`metrics`** | `top_k` | Integer | `3` | Default Top-K metric evaluation parameter. |
| | `ece_bins` | Integer | `10` | ECE calibration binning parameter. |
| | `auroc_mode` | String | `"macro"` | One-vs-Rest AUROC aggregation mode. |
| **`benchmark`** | `warmup_iterations`| Integer | `50` | Warm-up iterations for inference benchmarking. |
| | `benchmark_iterations`| Integer | `200` | Timed iterations for inference benchmarking. |
| | `use_fp16` | Boolean | `true` | Enables FP16 precision during latency benchmarking. |
| **`paths`** | `checkpoint_dir` | String | `"checkpoints"` | Directory path for saving model checkpoints. |
| | `log_dir` | String | `"logs"` | Directory path for logs and `metrics.csv`. |
| | `result_dir` | String | `"results"` | Directory path for JSON reports and visual PNG plots. |

---

## 🚩 Implemented Feature Flags vs. Not Implemented

| Feature Flag / Setting | Implementation Status | Implementation Details | Source Reference |
| :--- | :--- | :--- | :--- |
| **Automatic Mixed Precision (AMP)** | **Implemented** | Enabled when `mixed_precision.enabled: true` and `device.type == "cuda"`. Uses `torch.amp.autocast()` & `GradScaler`. | [`src/train.py:L261`](file:///d:/resnet%20crop%20detection/src/train.py#L261) |
| **Early Stopping** | **Implemented** | Enabled when `early_stopping.enabled: true`. Triggers when `val_loss` fails to decrease for `patience=10` consecutive epochs. | [`src/train.py:L202-L237`](file:///d:/resnet%20crop%20detection/src/train.py#L202-L237) |
| **Zero-Init Bottleneck Residuals** | **Implemented** | Enabled when `model.zero_init_residual: true`. Sets `bn3.weight=0.0` in all bottleneck residual blocks. | [`src/model.py:L183-L187`](file:///d:/resnet%20crop%20detection/src/model.py#L183-L187) |
| **Atomic Checkpointing** | **Implemented** | Writes to temporary file `.tmp.pt`, verifies loading, then calls `os.replace()` to replace target checkpoint. | [`src/train.py:L138-L167`](file:///d:/resnet%20crop%20detection/src/train.py#L138-L167) |
| **Deterministic Seed Mode** | **Implemented** | Enables seeds across `random`, `numpy`, `torch`, CUDA, and sets `cudnn.deterministic=True`. | [`src/utils.py:L72-L94`](file:///d:/resnet%20crop%20detection/src/utils.py#L72-L94) |
| **Distributed Data Parallel (DDP)**| **Not Found** | No `torch.distributed`, `torch.nn.parallel.DistributedDataParallel`, or multi-GPU process launching is implemented. | Codebase inspection |
| **Model Quantization (INT8)** | **Not Found** | No PyTorch post-training quantization (`torch.quantization`) or dynamic INT8 quantization is implemented. | Codebase inspection |
| **API Authentication / Rate Limits**| **Not Found** | No API keys, JWT bearer tokens, or HTTP request rate limiters are implemented in [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py). | [`deployment/app.py`](file:///d:/resnet%20crop%20detection/deployment/app.py) |
