# Configuration & Utilities Documentation

This document provides a reverse engineered technical specification of the system configuration files, utility scripts, reproducibility helpers, logging framework, and test suites implemented in [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml), [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py), and [`tests/`](file:///d:/resnet%20crop%20detection/tests/).

---

## ⚙️ Configuration File Schema (`configs/config.yaml`)

The system uses PyYAML to parse [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml):

```yaml
seed: 42                      # Global random seed
deterministic: true           # Enforce cuDNN deterministic algorithms
device: "auto"                # Device selection ("auto", "cuda", "cpu")

data:
  data_dir: "data"            # Base dataset folder
  train_dir: "data/train"     # Training split folder
  val_dir: "data/val"         # Validation split folder
  test_dir: "data/test"       # Test split folder
  num_classes: null           # Dynamically resolved if null
  img_size: 224               # Input spatial crop size
  resize_size: 256            # Pre-crop resize dimension
  batch_size: 32              # DataLoader batch size
  num_workers: 0              # Worker threads
  pin_memory: true            # Fast host-to-GPU memory transfer

model:
  name: "resnet50"
  pretrained: false           # Un-pretrained baseline
  dropout: 0.0                # Dropout rate (0.0 for standard ResNet)
  zero_init_residual: true    # Zero-initialize final BN in residual blocks

optimizer:
  name: "adamw"
  learning_rate: 0.0003       # Base learning rate (3e-4)
  weight_decay: 0.0001        # L2 regularization weight decay

scheduler:
  name: "cosine"
  min_lr: 0.000001            # Minimum learning rate (1e-6)

early_stopping:
  enabled: true
  patience: 10                # Epoch patience threshold
  monitor: "val_loss"
```

---

## 🛠️ Utility Functions (`src/utils.py`)

### 1. `set_seed(seed=42, deterministic=True)`
Ensures 100% full experiment reproducibility across Python, NumPy, PyTorch CPU, and PyTorch CUDA:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
```

### 2. `get_device(preferred_device="auto")`
Detects CUDA availability automatically, prints GPU properties (VRAM GB, CUDA version), and returns `torch.device("cuda:0")` or `torch.device("cpu")`.

### 3. `ExperimentLogger`
Dual-destination logger that formats stdout messages and appends structured metric records to CSV logs (`logs/metrics.csv` & `logs/training.log`).

---

## 🧪 Test & Verification Suites (`tests/`)

The repository contains milestone test suites powered by Python `unittest`:

- **[`test_milestone2.py`](file:///d:/resnet%20crop%20detection/tests/test_milestone2.py)**: Verifies data loading, image verification, transformations, and dataset integrity checks.
- **[`test_milestone3.py`](file:///d:/resnet%20crop%20detection/tests/test_milestone3.py)**: Tests custom ResNet-50 architecture initialization, forward pass shapes, parameter counts, and zero-residual initialization.
- **[`test_milestone4.py`](file:///d:/resnet%20crop%20detection/tests/test_milestone4.py)**: Validates training loop, AMP mixed precision scaler, AdamW step, learning rate scheduler, and early stopping.
- **[`test_milestone5.py`](file:///d:/resnet%20crop%20detection/tests/test_milestone5.py)**: Tests evaluation metrics (Macro F1, Top-1/Top-5, ECE calibration), checkpoint loading immutability, confusion matrix generation, and artifact generation.
