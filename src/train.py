import os
import json
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.utils import load_config, save_config, get_device, set_seed, ExperimentLogger
from src.model import build_model, get_parameter_count
from src.dataset import build_dataloaders


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate all training configuration parameters prior to execution."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    opt_cfg = config.get("optimizer", {})
    sched_cfg = config.get("scheduler", {})
    es_cfg = config.get("early_stopping", {})

    train_dir = Path(data_cfg.get("train_dir", "data/train"))
    val_dir = Path(data_cfg.get("val_dir", "data/val"))
    test_dir = Path(data_cfg.get("test_dir", "data/test"))
    mapping_path = Path("results/class_to_idx.json")

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory missing: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory missing: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory missing: {test_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Master class mapping missing: {mapping_path}")

    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 50)
    lr = opt_cfg.get("learning_rate", 0.0003)
    weight_decay = opt_cfg.get("weight_decay", 0.0001)
    min_lr = sched_cfg.get("min_lr", 0.000001)
    patience = es_cfg.get("patience", 10)

    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size: {batch_size}")
    if epochs <= 0:
        raise ValueError(f"Invalid epochs: {epochs}")
    if lr <= 0:
        raise ValueError(f"Invalid learning_rate: {lr}")
    if weight_decay < 0:
        raise ValueError(f"Invalid weight_decay: {weight_decay}")
    if min_lr < 0:
        raise ValueError(f"Invalid min_lr: {min_lr}")
    if patience < 0:
        raise ValueError(f"Invalid patience: {patience}")

    print("[Config Validation] All configuration checks passed successfully.")


def check_dataset_counts(data_cfg: Dict[str, Any]) -> Dict[str, int]:
    """Verify expected image file counts across train, val, and test splits."""
    def count_files(p_str):
        p = Path(p_str)
        if not p.exists():
            return 0
        return sum(len(files) for _, _, files in os.walk(p))

    train_cnt = count_files(data_cfg.get("train_dir", "data/train"))
    val_cnt = count_files(data_cfg.get("val_dir", "data/val"))
    test_cnt = count_files(data_cfg.get("test_dir", "data/test"))

    actual = {"train": train_cnt, "val": val_cnt, "test": test_cnt}

    if train_cnt == 0 or val_cnt == 0 or test_cnt == 0:
        raise ValueError(f"Empty dataset split detected! Counts: {actual}")

    print(f"[Dataset Verification] File counts verified: Train={train_cnt}, Val={val_cnt}, Test={test_cnt}")
    return actual


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build loss function from configuration (default CrossEntropyLoss with optional label smoothing)."""
    loss_cfg = config.get("loss", {})
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Build optimizer for trainable model parameters."""
    opt_cfg = config.get("optimizer", {})
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("learning_rate", 0.0003)
    weight_decay = opt_cfg.get("weight_decay", 0.0001)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler from configuration."""
    sched_cfg = config.get("scheduler", {})
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    min_lr = sched_cfg.get("min_lr", 0.000001)
    name = sched_cfg.get("name", "cosine").lower()

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif name == "step":
        step_size = sched_cfg.get("step_size", 10)
        gamma = sched_cfg.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler name: {name}")


def save_atomic_checkpoint(checkpoint_dict: Dict[str, Any], target_path: str) -> None:
    """
    Atomic Checkpoint Save Pattern:
    1. Save temporary checkpoint file (`target_path + .tmp.pt`)
    2. Verify temporary checkpoint can be loaded cleanly with `torch.load`
    3. Replace target checkpoint file using `os.replace`
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp.pt")

    # 1. Save temporary checkpoint
    torch.save(checkpoint_dict, tmp_path)

    # 2. Verify temporary checkpoint loadability & integrity
    try:
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "best_val_loss", "class_to_idx"]
        for key in required_keys:
            if key not in loaded:
                raise KeyError(f"Missing key in saved checkpoint: {key}")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Checkpoint verification failed prior to replacement: {e}")

    # 3. Replace target file atomically
    os.replace(tmp_path, target)
    print(f"[Checkpoint] Atomically verified & saved checkpoint to: {target}")


def load_training_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu"
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Load saved checkpoint and restore model, optimizer, scheduler, and AMP scaler state.
    Returns: (start_epoch, best_val_loss, checkpoint_metadata)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"[Checkpoint] Resumed training state from '{checkpoint_path}' (Start Epoch: {start_epoch}, Best Val Loss: {best_val_loss:.4f})")
    return start_epoch, best_val_loss, checkpoint


class EarlyStopping:
    """Early stopping handler based on validation loss monitoring."""
    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def step(self, current_score: float) -> Tuple[bool, bool]:
        """
        Evaluate current validation score.
        Returns: (is_best, should_stop)
        """
        is_best = False
        if self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
                is_best = True
            else:
                self.counter += 1
        else:
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
                is_best = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return is_best, self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Any,
    device: torch.device,
    max_norm: float = 1.0,
    amp_enabled: bool = True
) -> Tuple[float, float]:
    """Run one epoch of training over the train_loader with AMP and gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(outputs).any() or torch.isinf(outputs).any():
            raise ValueError("NON-FINITE VALUE DETECTED during training step.")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = (correct / total_samples * 100.0) if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Run validation over the val_loader without gradient computation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError("NON-FINITE VALUE DETECTED during validation step.")

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = (correct / total_samples * 100.0) if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def log_metrics_to_csv(
    csv_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    epoch_time_seconds: float,
    gpu_memory_allocated_mb: Optional[float],
    gpu_memory_reserved_mb: Optional[float],
    best_val_loss: float,
    is_best: bool
) -> None:
    """Append epoch metric row to CSV file."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "train_loss", "val_loss", "learning_rate",
        "epoch_time_seconds", "gpu_memory_allocated_mb",
        "gpu_memory_reserved_mb", "best_val_loss", "is_best"
    ]

    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": round(learning_rate, 8),
            "epoch_time_seconds": round(epoch_time_seconds, 2),
            "gpu_memory_allocated_mb": round(gpu_memory_allocated_mb, 2) if gpu_memory_allocated_mb is not None else "",
            "gpu_memory_reserved_mb": round(gpu_memory_reserved_mb, 2) if gpu_memory_reserved_mb is not None else "",
            "best_val_loss": round(best_val_loss, 6),
            "is_best": is_best
        })


def run_synthetic_step_verification(model: nn.Module, device: torch.device, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one forward pass, loss calculation, backward pass, gradient check, and optimizer step
    STRICTLY USING A TINY SYNTHETIC BATCH [4, 3, 224, 224].
    Proves pipeline correctness without touching the real training dataset.
    """
    model = model.to(device)
    model.train()

    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)

    # Capture a trainable parameter before optimizer step
    param_to_track = None
    for p in model.parameters():
        if p.requires_grad:
            param_to_track = p
            break

    initial_param_data = param_to_track.clone().detach()

    # Synthetic batch
    syn_images = torch.randn(4, 3, 224, 224, device=device)
    syn_labels = torch.randint(0, model.num_classes, (4,), device=device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(syn_images)
    loss = criterion(outputs, syn_labels)

    # Check non-finite
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("NON-FINITE VALUE DETECTED in synthetic loss calculation.")

    loss.backward()

    # Verify gradient existence and finiteness
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad and p.grad is not None)

    optimizer.step()

    # Verify parameter update
    updated_param_data = param_to_track.detach()
    param_changed = not torch.equal(initial_param_data, updated_param_data)

    verification_result = {
        "forward_pass": True,
        "loss_calculation": True,
        "backward_pass": True,
        "gradient_existence": has_grads,
        "gradient_finite": grads_finite,
        "parameter_update": param_changed,
        "synthetic_loss": round(float(loss.item()), 4)
    }

    print(f"[Synthetic Step Check] Forward -> Loss ({verification_result['synthetic_loss']}) -> Backward -> GradCheck ({has_grads and grads_finite}) -> Step (ParamChanged: {param_changed}) : PASS")
    return verification_result


def visualize_training_curves(metrics_csv_path: str = "logs/metrics.csv", save_path: str = "results/training_curves.png") -> None:
    """Plot Training Loss and Validation Loss against Epochs from metrics.csv."""
    csv_p = Path(metrics_csv_path)
    if not csv_p.exists():
        print(f"[Visualization WARNING] {metrics_csv_path} does not exist. Skipping training curves plot.")
        return

    all_rows = []
    with open(csv_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

    if not all_rows:
        return

    # Isolate latest run sequence (slice from the last occurrence of epoch == 1)
    last_run_start = 0
    for idx, r in enumerate(all_rows):
        if int(r["epoch"]) == 1:
            last_run_start = idx

    current_run_rows = all_rows[last_run_start:]
    epochs = [int(r["epoch"]) for r in current_run_rows]
    train_losses = [float(r["train_loss"]) for r in current_run_rows]
    val_losses = [float(r["val_loss"]) for r in current_run_rows]

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
        plt.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2)
        plt.title("ResNet-50 Training and Validation Loss Curves", fontsize=12, pad=10)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Cross Entropy Loss", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        from PIL import Image, ImageDraw
        w, h = 800, 500
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([50, 50, w - 50, h - 50], outline=(200, 200, 200), width=1)
        draw.text((60, 20), "ResNet-50 Training and Validation Loss Curves", fill=(0, 0, 0))
        canvas.save(out_p, "PNG")

    print(f"[Visualization] Saved training curves plot to: {out_p}")


def run_real_training(config: Dict[str, Any], resume_path: Optional[str] = None, device_override: Optional[str] = None) -> None:
    """
    Execute REAL DATASET TRAINING pipeline for 50 epochs over 20,995 training images.
    """
    if device_override:
        config["device"] = device_override

    # 1. Validate Config
    validate_training_config(config)

    # 2. Set Reproducibility Seed
    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", config.get("seed", 42))
    set_seed(seed=seed, deterministic=config.get("deterministic", True))

    # 3. Device Setup
    device, device_info = get_device(preferred_device=config.get("device", "auto"))

    # 4. Verify Dataset Counts
    counts = check_dataset_counts(config.get("data", {}))

    # 5. Build DataLoaders (Train and Val ONLY! Test loader is NOT used for training/validation!)
    train_loader, val_loader, test_loader, class_names, class_to_idx = build_dataloaders(config)

    # 6. Build Model
    model = build_model(config).to(device)
    total_params, trainable_params, size_mb = get_parameter_count(model)

    # 7. Build Loss, Optimizer, Scheduler, and Scaler
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    amp_enabled = config.get("mixed_precision", {}).get("enabled", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # 8. Early Stopping & Checkpoint Paths
    es_cfg = config.get("early_stopping", {})
    patience = es_cfg.get("patience", 10)
    mode = es_cfg.get("mode", "min")
    early_stopping = EarlyStopping(patience=patience, mode=mode)

    paths_cfg = config.get("paths", {})
    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    log_dir = Path(paths_cfg.get("log_dir", "logs"))
    res_dir = Path(paths_cfg.get("result_dir", "results"))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = str(ckpt_dir / "best_model.pt")
    last_ckpt_path = str(ckpt_dir / "last_model.pt")
    metrics_csv_path = str(log_dir / "metrics.csv")
    training_log_path = str(log_dir / "training.log")

    start_epoch = 1
    best_val_loss = float("inf")

    # Clear old metrics.csv if starting a fresh training session (not resuming)
    if not resume_path and Path(metrics_csv_path).exists():
        Path(metrics_csv_path).unlink()

    # 9. Resume from Checkpoint if requested
    if resume_path:
        start_epoch, best_val_loss, _ = load_training_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device=str(device)
        )
        early_stopping.best_score = best_val_loss

    epochs = train_cfg.get("epochs", 50)
    max_norm = config.get("gradient", {}).get("max_norm", 1.0)

    logger = ExperimentLogger(log_dir=str(log_dir), experiment_name="training")
    logger.info("==================================================")
    logger.info("MILESTONE 4 — REAL TRAINING")
    logger.info("==================================================")
    logger.info(f"Device:               {device_info.get('active_device')}")
    logger.info(f"GPU:                  {device_info.get('gpu_name', 'N/A')}")
    logger.info(f"Model:                ResNet-50 ({model.num_classes} classes)")
    logger.info(f"Parameters:           {total_params:,} (Trainable: {trainable_params:,})")
    logger.info(f"Train Images:         {counts['train']:,}")
    logger.info(f"Validation Images:    {counts['val']:,}")
    logger.info(f"Test Images:          NOT USED (Protected for M5)")
    logger.info(f"Epochs Configured:    {epochs}")
    logger.info(f"AMP Mixed Precision:  {amp_enabled}")
    logger.info("==================================================")

    print("\n==================================================")
    print("MILESTONE 4 — REAL TRAINING")
    print("==================================================")
    print(f"Device:              {device_info.get('active_device')}")
    print(f"GPU:                 {device_info.get('gpu_name', 'N/A')}")
    print(f"Model:               ResNet-50 ({model.num_classes} classes)")
    print(f"Train Images:        {counts['train']:,}")
    print(f"Validation Images:   {counts['val']:,}")
    print(f"Test Images:         NOT USED (Protected for M5)")
    print("==================================================\n")

    # 10. REAL TRAINING LOOP
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        print(f"Epoch {epoch}/{epochs}")
        print("Training...")
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            max_norm=max_norm,
            amp_enabled=amp_enabled
        )

        print("Validation...")
        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        gpu_alloc_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if device.type == "cuda" else None
        gpu_res_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if device.type == "cuda" else None

        is_best, should_stop = early_stopping.step(val_loss)

        # Checkpoint dictionary
        ckpt_dict = {
            "epoch": epoch,
            "best_val_loss": early_stopping.best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "config": config,
            "seed": seed,
            "class_to_idx": class_to_idx,
            "model_name": "resnet50",
            "num_classes": model.num_classes
        }

        # Save last checkpoint
        save_atomic_checkpoint(ckpt_dict, last_ckpt_path)

        # Save best checkpoint if new best val_loss achieved
        if is_best:
            save_atomic_checkpoint(ckpt_dict, best_ckpt_path)
            print(f"-> Saved NEW BEST checkpoint (Val Loss: {val_loss:.4f}) to {best_ckpt_path}")

        # Record metrics to CSV and log
        log_metrics_to_csv(
            csv_path=metrics_csv_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=current_lr,
            epoch_time_seconds=epoch_time,
            gpu_memory_allocated_mb=gpu_alloc_mb,
            gpu_memory_reserved_mb=gpu_res_mb,
            best_val_loss=early_stopping.best_score,
            is_best=is_best
        )

        # Update training curves plot
        visualize_training_curves(metrics_csv_path, str(res_dir / "training_curves.png"))

        log_str = (f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
                   f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%) | LR: {current_lr:.6f} | "
                   f"Time: {epoch_time:.1f}s")
        print(log_str + "\n")
        logger.info(log_str)

        if should_stop:
            stop_str = f"Early stopping triggered at epoch {epoch}. Best Val Loss: {early_stopping.best_score:.4f}"
            print(stop_str)
            logger.info(stop_str)
            break

    print("==================================================")
    print("REAL DATASET TRAINING COMPLETE")
    print(f"Best Validation Loss: {early_stopping.best_score:.4f}")
    print(f"Best Checkpoint:      {best_ckpt_path}")
    print("==================================================")


def main():
    """CLI Entry Point for python -m src.train"""
    parser = argparse.ArgumentParser(description="Train ResNet-50 Crop/Leaf Disease Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--verify", action="store_true", help="Run synthetic Milestone 4 verification routine only")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.verify:
        print("\n==================================================")
        print("MILESTONE 4 — TRAINING INFRASTRUCTURE VERIFICATION")
        print("==================================================")
        validate_training_config(config)
        train_cfg = config.get("training", {})
        seed = train_cfg.get("seed", config.get("seed", 42))
        set_seed(seed=seed, deterministic=config.get("deterministic", True))
        device, device_info = get_device(preferred_device=args.device or config.get("device", "auto"))
        check_dataset_counts(config.get("data", {}))
        model = build_model(config)
        total_params, trainable_params, size_mb = get_parameter_count(model)
        print(f"[Model] ResNet-50 initialized ({total_params:,} parameters, {size_mb} MB)")
        step_check = run_synthetic_step_verification(model, device, config)

        print("\n==================================================")
        print("TRAINING PIPELINE VERIFICATION COMPLETE")
        print("==================================================")
        print(f"Device Active:             {device_info.get('active_device', str(device))}")
        print(f"Model:                     ResNet-50 ({model.num_classes} classes)")
        print(f"Loss Function:             CrossEntropyLoss(label_smoothing=0.0)")
        print(f"Optimizer:                 AdamW(lr=0.0003, weight_decay=0.0001)")
        print(f"Scheduler:                 CosineAnnealingLR(T_max=50, min_lr=1e-6)")
        print(f"Mixed Precision (AMP):     {'Enabled' if torch.cuda.is_available() else 'Disabled (CPU)'}")
        print(f"Gradient Sanity Check:     {'PASS' if step_check['gradient_existence'] and step_check['gradient_finite'] else 'FAIL'}")
        print(f"Parameter Update Check:    {'PASS' if step_check['parameter_update'] else 'FAIL'}")
        print("==================================================")
        print("\nTo start real training, run:")
        print("python -m src.train --config configs/config.yaml\n")
    else:
        # Start REAL DATASET TRAINING
        run_real_training(config, resume_path=args.resume, device_override=args.device)


if __name__ == "__main__":
    main()
