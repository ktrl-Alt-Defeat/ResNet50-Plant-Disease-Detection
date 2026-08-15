import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

from src.model import build_model
from src.utils import load_config
from deployment.preprocessing import get_inference_transforms, preprocess_image_bytes


class ModelEngine:
    """
    Singleton Inference Engine for ResNet-50 Crop Disease Classifier.
    Handles model initialization, checkpoint loading, device selection, and inference execution.
    """

    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt", mapping_path: str = "results/class_to_idx.json"):
        self.checkpoint_path = Path(checkpoint_path)
        self.mapping_path = Path(mapping_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = None
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.num_classes: int = 0
        self.transform = get_inference_transforms()
        self.is_loaded: bool = False

    def load_model(self) -> None:
        """
        Load weights, set eval mode, and build class mappings.
        Executed ONCE at FastAPI application startup.
        """
        if self.is_loaded:
            return

        # 1. Resolve Class Mapping (Prefer results/class_to_idx.json as authoritative)
        if self.mapping_path.exists():
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.class_to_idx = json.load(f)
            print(f"[ModelEngine] Loaded class mapping from authoritative file: {self.mapping_path}")
        else:
            self.class_to_idx = {}

        # 2. Check Checkpoint Existence
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        print(f"[ModelEngine] Loading checkpoint: {self.checkpoint_path} on device: {self.device}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Fallback to checkpoint mapping if results/class_to_idx.json was missing
        if not self.class_to_idx and "class_to_idx" in checkpoint:
            self.class_to_idx = checkpoint["class_to_idx"]

        if not self.class_to_idx:
            raise KeyError("Class mapping could not be found in results/class_to_idx.json or checkpoint.")

        self.num_classes = len(self.class_to_idx)
        self.idx_to_class = {int(idx): name for name, idx in self.class_to_idx.items()}

        # 3. Build Model Instance & Load Weights
        config = load_config()
        self.model = build_model(config, num_classes=self.num_classes)
        
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        print(f"[ModelEngine] ResNet-50 loaded successfully! ({self.num_classes} classes, Device: {self.device})")

    @torch.inference_mode()
    def predict(self, file_bytes: bytes, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform model inference on raw image bytes.
        
        Args:
            file_bytes: Raw image file bytes from HTTP request
            top_k: Number of top probability predictions to return (default 5)
            
        Returns:
            Dictionary containing predicted_class, confidence, confidence_percentage, and top_5_predictions
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model engine is not initialized or model failed to load.")

        # Preprocess Image
        input_tensor = preprocess_image_bytes(file_bytes, self.transform).to(self.device)

        # Forward Pass & Softmax
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # Compute Top-K predictions
        k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.idx_to_class.get(idx, f"unknown_class_{idx}")
            top_predictions.append({
                "class": class_name,
                "confidence": round(float(prob), 4)
            })

        top_1_class = top_predictions[0]["class"]
        top_1_conf = top_predictions[0]["confidence"]
        top_1_pct = round(top_1_conf * 100.0, 2)

        return {
            "predicted_class": top_1_class,
            "confidence": top_1_conf,
            "confidence_percentage": top_1_pct,
            "top_5_predictions": top_predictions
        }


# Global Singleton Instance
model_engine = ModelEngine()
