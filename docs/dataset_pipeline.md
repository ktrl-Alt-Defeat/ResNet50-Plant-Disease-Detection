# Dataset & Validation Pipeline Documentation

This document provides a reverse engineered technical specification of the dataset loader, image augmentations, validation engine, and leakage detection mechanics implemented in [`src/dataset.py`](file:///d:/resnet%20crop%20detection/src/dataset.py) and [`src/dataset_validation.py`](file:///d:/resnet%20crop%20detection/src/dataset_validation.py).

---

## 📦 Dataset Subsystem (`src/dataset.py`)

### 1. `PlantDiseaseDataset` (PyTorch `Dataset`)
- **Inheritance**: `torch.utils.data.Dataset`
- **Supported File Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- **Class Discovery**: Scans root directory subfolders alphabetically to establish class index mappings.
- **Image Integrity Verification**:
  ```python
  with Image.open(img_path) as img:
      img.verify()
  ```
  Corrupted or 0-byte images are caught and safely reported.
- **Color Format Normalization**: Forces conversion to 3-channel RGB (`img.convert("RGB")`) for grayscale, RGBA, or palette images.

---

## 🎨 Image Transformation & Data Augmentations

The transformation pipelines adhere strictly to ImageNet normalization standards:
- **Mean**: `[0.485, 0.456, 0.406]`
- **Std**: `[0.229, 0.224, 0.225]`

### Training Transformation Pipeline (`get_transforms(is_train=True)`)
1. `transforms.Resize(256)`: Resizes image short edge to 256 pixels.
2. `transforms.RandomResizedCrop(224, scale=(0.8, 1.0))`: Random cropping with scale constraints.
3. `transforms.RandomHorizontalFlip(p=0.5)`: Flips image horizontally.
4. `transforms.RandomVerticalFlip(p=0.2)`: Flips image vertically.
5. `transforms.RandomRotation(degrees=15)`: Rotates image within $\pm 15^\circ$.
6. `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: Photometric jittering.
7. `transforms.ToTensor()`: Converts PIL image $[0, 255]$ to FloatTensor $[0.0, 1.0]$.
8. `transforms.Normalize(mean, std)`: Standardizes channel distributions.

### Validation & Test Pipeline (`get_transforms(is_train=False)`)
1. `transforms.Resize(256)`
2. `transforms.CenterCrop(224)`
3. `transforms.ToTensor()`
4. `transforms.Normalize(mean, std)`

---

## 🔍 Dataset Validation & Leakage Prevention (`src/dataset_validation.py`)

The validation pipeline performs strict automated checks prior to model training:

### 1. File Integrity & Corrupt Image Detector
- Verifies header magic bytes and image dimensions.
- Flags unreadable or truncated image files across train, validation, and test directories.

### 2. Cross-Split Dataset Leakage Detection
- Computes **MD5** and **SHA-256** cryptographic hashes for every image file in `train`, `val`, and `test` splits.
- If identical hashes exist across splits (e.g. an image present in both `train` and `test`), leakage is flagged and documented in `results/dataset_validation_report.json`.

### 3. Class Imbalance Analysis
- Calculates class distributions, min/max sample counts per class, mean, median, and standard deviation.
- Outputs distribution summary to `results/dataset_class_distribution.csv`.
