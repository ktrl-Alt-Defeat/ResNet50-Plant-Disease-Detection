import io
from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms

# ImageNet default normalization constants matching evaluation pipeline
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transforms(img_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    """
    Construct evaluation/inference image transformation pipeline.
    Matches evaluation pipeline: Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize()
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_and_validate_image(file_bytes: bytes) -> Image.Image:
    """
    Safely load, verify, and convert raw byte data into a PIL RGB Image.
    Raises ValueError if content is corrupt, empty, or unreadable as an image.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        # First verify image integrity
        img_stream = io.BytesIO(file_bytes)
        with Image.open(img_stream) as img:
            img.verify()

        # Reopen after verify() (PIL requirement)
        img_stream.seek(0)
        img = Image.open(img_stream)
        img.load()

        # Validate non-zero dimensions
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        # Ensure 3-channel RGB format
        return img.convert("RGB")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid or corrupted image format ({type(e).__name__}: {str(e)})")


def preprocess_image_bytes(file_bytes: bytes, transform: transforms.Compose) -> torch.Tensor:
    """
    Process raw image bytes through validation and inference transform.
    Returns:
        torch.Tensor of shape [1, 3, 224, 224] ready for model input.
    """
    image = load_and_validate_image(file_bytes)
    tensor = transform(image)
    return tensor.unsqueeze(0)
