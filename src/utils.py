import os
import random
import logging
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_device(preferred_device: str = "auto") -> Tuple[torch.device, Dict[str, Any]]:
    """
    Detect CUDA GPU availability automatically and return torch device along with metadata.
    
    Args:
        preferred_device: "auto", "cuda", or "cpu"
        
    Returns:
        device: torch.device instance
        info: Dictionary containing device metadata (name, memory, CUDA version, etc.)
    """
    device_info = {
        "requested": preferred_device,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": None,
        "total_memory_gb": None,
    }

    if preferred_device.lower() in ["auto", "cuda"] and torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        device_info["gpu_name"] = props.name
        device_info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        device_info["active_device"] = f"CUDA:0 ({props.name})"
        print(f"[Device Setup] Using GPU: {props.name} ({device_info['total_memory_gb']} GB VRAM)")
        print(f"[Device Setup] CUDA Version: {device_info['cuda_version']}")
    else:
        device = torch.device("cpu")
        if preferred_device.lower() == "cuda" and not torch.cuda.is_available():
            print("[Device Setup] WARNING: CUDA requested but not available. Falling back to CPU.")
        else:
            print("[Device Setup] Using CPU.")
        device_info["active_device"] = "CPU"

    return device, device_info


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed across all libraries for full reproducibility.
    
    Args:
        seed: Integer seed
        deterministic: If True, forces cuDNN deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[Seed] Reproducibility enabled: seed={seed} (cuDNN deterministic=True)")
    else:
        torch.backends.cudnn.benchmark = True
        print(f"[Seed] Performance mode enabled: seed={seed} (cuDNN benchmark=True)")


def save_checkpoint(
    state_dict: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint_latest.pth",
    best_filename: str = "best_model.pth"
) -> str:
    """
    Save experiment checkpoint with rich metadata.
    
    State dict should contain:
      - model_state
      - optimizer_state
      - scheduler_state (optional)
      - epoch
      - best_macro_f1
      - class_names
      - config
      - seed
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    save_path = path / filename
    torch.save(state_dict, save_path)
    
    if is_best:
        best_path = path / best_filename
        torch.save(state_dict, best_path)
        print(f"[Checkpoint] Best model saved to: {best_path}")
        return str(best_path)
    
    return str(save_path)


def load_checkpoint(checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load model checkpoint with metadata."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


class ExperimentLogger:
    """Utility class to log stdout messages and record metric histories into CSV files."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.csv_file = self.log_dir / "metrics.csv"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        self._csv_initialized = False

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def log_metrics(self, epoch: int, metrics: Dict[str, float], step_type: str = "train") -> None:
        """Append epoch metrics to CSV log file."""
        row_data = {"epoch": epoch, "step": step_type, **metrics}
        fieldnames = list(row_data.keys())
        
        file_exists = self.csv_file.exists()
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row_data)
