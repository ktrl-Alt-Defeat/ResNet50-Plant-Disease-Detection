import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont

from src.utils import load_config, set_seed

# Default ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MappedImageFolder(datasets.ImageFolder):
    """
    Subclass of torchvision.datasets.ImageFolder that enforces a master class_to_idx mapping.
    This guarantees consistent target indices across train, val, and test splits even when
    validation or test splits have missing classes.
    """
    def __init__(self, root: str, transform=None, master_class_to_idx: Optional[Dict[str, int]] = None):
        self.master_class_to_idx = master_class_to_idx
        super().__init__(root=root, transform=transform)
        if master_class_to_idx is not None:
            self.class_to_idx = master_class_to_idx
            self.classes = sorted(list(master_class_to_idx.keys()))
            new_samples = []
            for path, _ in self.samples:
                class_name = Path(path).parent.name
                if class_name in master_class_to_idx:
                    new_samples.append((path, master_class_to_idx[class_name]))
            self.samples = new_samples
            self.targets = [s[1] for s in new_samples]


def infer_num_classes(train_dir: str = "data/train") -> Tuple[int, List[str]]:
    """
    Dynamically scan dataset train directory to infer sorted class names and total class count.
    """
    path = Path(train_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory '{train_dir}' does not exist.")

    class_names = sorted([d.name for d in path.iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise ValueError(f"No class subdirectories found in dataset path: {train_dir}")

    return len(class_names), class_names


def get_class_names(train_dir: str = "data/train") -> List[str]:
    """Get sorted list of class names from train directory."""
    _, class_names = infer_num_classes(train_dir)
    return class_names


def get_class_mapping(train_dir: str = "data/train", save_path: str = "results/class_to_idx.json") -> Dict[str, int]:
    """
    Construct deterministic class_to_idx mapping from train directory and save as JSON artifact.
    """
    class_names = get_class_names(train_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=4)
        print(f"[Dataset] Class mapping saved to: {out_p}")

    return class_to_idx


def get_transforms(config: Optional[Dict[str, Any]] = None) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Construct training (with augmentations) and validation/test transforms.
    
    Args:
        config: Configuration dictionary (or None for default settings)
        
    Returns:
        train_transforms, val_test_transforms
    """
    if config is None:
        config = {}
    
    data_cfg = config.get("data", {})
    img_size = data_cfg.get("img_size", 224)
    resize_size = data_cfg.get("resize_size", 256)
    aug_cfg = data_cfg.get("augmentation", {})
    norm_cfg = data_cfg.get("normalization", {})

    crop_scale = tuple(aug_cfg.get("random_crop_scale", [0.8, 1.0]))
    h_flip_prob = aug_cfg.get("horizontal_flip_prob", 0.5)
    v_flip_prob = aug_cfg.get("vertical_flip_prob", 0.2)
    rotation_deg = aug_cfg.get("rotation_degrees", 15)
    cj_val = aug_cfg.get("color_jitter", 0.2)

    mean = norm_cfg.get("mean", IMAGENET_MEAN)
    std = norm_cfg.get("std", IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=h_flip_prob),
        transforms.RandomVerticalFlip(p=v_flip_prob),
        transforms.RandomRotation(degrees=rotation_deg),
        transforms.ColorJitter(brightness=cj_val, contrast=cj_val, saturation=cj_val),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform


def create_datasets(config: Dict[str, Any]) -> Tuple[Dict[str, MappedImageFolder], List[str], Dict[str, int]]:
    """
    Build PyTorch Datasets for train, val, and test splits with master class mapping.
    """
    data_cfg = config.get("data", {})
    train_dir = data_cfg.get("train_dir", "data/train")
    val_dir = data_cfg.get("val_dir", "data/val")
    test_dir = data_cfg.get("test_dir", "data/test")

    class_to_idx = get_class_mapping(train_dir)
    class_names = sorted(list(class_to_idx.keys()))

    train_tf, val_tf = get_transforms(config)
    datasets_dict = {}

    if Path(train_dir).exists():
        datasets_dict["train"] = MappedImageFolder(train_dir, transform=train_tf, master_class_to_idx=class_to_idx)

    if Path(val_dir).exists():
        datasets_dict["val"] = MappedImageFolder(val_dir, transform=val_tf, master_class_to_idx=class_to_idx)

    if Path(test_dir).exists():
        datasets_dict["test"] = MappedImageFolder(test_dir, transform=val_tf, master_class_to_idx=class_to_idx)

    return datasets_dict, class_names, class_to_idx


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Construct PyTorch DataLoaders for train, val, and test splits.
    
    Returns:
        train_loader, val_loader, test_loader, class_names, class_to_idx
    """
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    drop_last = data_cfg.get("drop_last", False)
    
    # Configure pin_memory dynamically based on CUDA availability
    pin_memory = data_cfg.get("pin_memory", True) if torch.cuda.is_available() else False

    datasets_dict, class_names, class_to_idx = create_datasets(config)

    train_loader = DataLoader(
        datasets_dict["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    ) if "train" in datasets_dict else None

    val_loader = DataLoader(
        datasets_dict["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    ) if "val" in datasets_dict else None

    test_loader = DataLoader(
        datasets_dict["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    ) if "test" in datasets_dict else None

    return train_loader, val_loader, test_loader, class_names, class_to_idx


def verify_dataloader_batches(
    train_loader: Optional[DataLoader],
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader]
) -> Dict[str, Any]:
    """
    Fetch exactly one batch from each DataLoader and verify tensor shapes, dtypes, and value ranges.
    Prints formatted terminal output as required by Phase 12.
    """
    verification_results = {}

    print("========================================")
    print("DATALOADER VERIFICATION")
    print("========================================")

    for split_name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
        print(f"\n{split_name}:")
        if loader is None:
            print("  Loader: NOT AVAILABLE")
            continue
        try:
            images, labels = next(iter(loader))
            v_info = {
                "img_shape": list(images.shape),
                "lbl_shape": list(labels.shape),
                "img_dtype": str(images.dtype),
                "lbl_dtype": str(labels.dtype),
                "min_val": round(float(images.min()), 4),
                "max_val": round(float(images.max()), 4)
            }
            verification_results[split_name.lower()] = v_info
            print(f"Images: shape={v_info['img_shape']}, dtype={v_info['img_dtype']}, range=[{v_info['min_val']}, {v_info['max_val']}]")
            print(f"Labels: shape={v_info['lbl_shape']}, dtype={v_info['lbl_dtype']}")
        except Exception as e:
            print(f"  FAIL: Could not load batch ({e})")
            verification_results[split_name.lower()] = {"error": str(e)}

    print("\n========================================")
    return verification_results


def visualize_dataset_samples(
    dataloader: DataLoader,
    class_to_idx: Dict[str, int],
    save_path: str = "results/dataset_samples.png",
    num_samples: int = 16,
    seed: int = 42
) -> str:
    """
    Display and save a grid of representative training images with class names using PIL.
    Applies inverse ImageNet normalization before saving.
    """
    set_seed(seed)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    images, labels = next(iter(dataloader))
    num_samples = min(num_samples, len(images))

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    tile_size = 224
    header_height = 32
    padding = 8
    
    canvas_w = cols * tile_size + (cols + 1) * padding
    canvas_h = rows * (tile_size + header_height) + (rows + 1) * padding
    
    # Create canvas (light gray background)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(240, 242, 245))
    draw = ImageDraw.Draw(canvas)

    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    for i in range(num_samples):
        r = i // cols
        c = i % cols
        
        x = padding + c * (tile_size + padding)
        y = padding + r * (tile_size + header_height + padding)

        # Inverse normalize: img = img * std + mean
        img_tensor = images[i].numpy().transpose(1, 2, 0)
        img_unnorm = img_tensor * std + mean
        img_unnorm = np.clip(img_unnorm * 255.0, 0, 255).astype(np.uint8)

        tile_img = Image.fromarray(img_unnorm).resize((tile_size, tile_size))
        
        # Paste tile image
        canvas.paste(tile_img, (x, y + header_height))

        # Draw text header box
        draw.rectangle([x, y, x + tile_size, y + header_height], fill=(30, 41, 59))
        label_idx = labels[i].item()
        class_name = idx_to_class.get(label_idx, f"Class {label_idx}")
        if len(class_name) > 22:
            class_name = class_name[:20] + "..."

        draw.text((x + 6, y + 8), class_name, fill=(255, 255, 255))

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p, "PNG")

    print(f"[Dataset] Sample visualization saved to: {out_p}")
    return str(out_p)


def main():
    """CLI entry point for python -m src.dataset"""
    config = load_config()
    set_seed(config.get("seed", 42), config.get("deterministic", True))

    print("[Dataset CLI Test] Building DataLoaders...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = build_dataloaders(config)

    print(f"[Dataset CLI Test] Total Master Classes: {len(class_names)}")
    print(f"[Dataset CLI Test] Train Loader Batches: {len(train_loader) if train_loader else 0}")
    print(f"[Dataset CLI Test] Val Loader Batches:   {len(val_loader) if val_loader else 0}")
    print(f"[Dataset CLI Test] Test Loader Batches:  {len(test_loader) if test_loader else 0}")

    print("\n[Dataset CLI Test] Verifying Batches...")
    verify_results = verify_dataloader_batches(train_loader, val_loader, test_loader)

    print("\n[Dataset CLI Test] Generating Sample Grid Visualization...")
    if train_loader:
        visualize_dataset_samples(train_loader, class_to_idx, save_path="results/dataset_samples.png")

    print("\n[Dataset CLI Test] DataLoader pipeline successfully verified!")


if __name__ == "__main__":
    main()
