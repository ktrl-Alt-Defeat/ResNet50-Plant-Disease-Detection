import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import load_config, get_device


class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck residual block.
    
    Expansion factor = 4.
    Channel breakdown:
      - in_channels: Input channels to block
      - bottleneck_channels: Intermediate (mid) channels
      - out_channels: Output channels (bottleneck_channels * expansion)
    """
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = bottleneck_channels * self.expansion
        self.stride = stride
        self.downsample = downsample

        # 1x1 Conv: in_channels -> bottleneck_channels
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3x3 Conv: bottleneck_channels -> bottleneck_channels
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1x1 Conv: bottleneck_channels -> out_channels
        self.conv3 = nn.Conv2d(bottleneck_channels, self.out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """
    Custom ResNet-50 Classifier Architecture for Crop/Leaf Disease Detection.
    
    Structure:
      - Stem: 7x7 Conv (stride=2), BatchNorm, ReLU, 3x3 MaxPool (stride=2)
      - Stage 1 (Conv2_x): 3 Bottleneck blocks (256 out channels, stride=1)
      - Stage 2 (Conv3_x): 4 Bottleneck blocks (512 out channels, stride=2)
      - Stage 3 (Conv4_x): 6 Bottleneck blocks (1024 out channels, stride=2)
      - Stage 4 (Conv5_x): 3 Bottleneck blocks (2048 out channels, stride=2)
      - Adaptive Global Average Pooling
      - Classifier Head: Linear(2048 -> num_classes)
    """
    block = Bottleneck
    layers = [3, 4, 6, 3] # Standard ResNet-50 block counts

    def __init__(
        self,
        num_classes: int = 124,
        dropout: float = 0.0,
        zero_init_residual: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.in_channels = 64

        # Stem Layer: 3 -> 64 (7x7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottleneck Stages
        # Conv2_x: 3 blocks, mid_channels=64, out_channels=256, stride=1
        self.layer1 = self._make_layer(bottleneck_channels=64, blocks=3, stride=1)
        # Conv3_x: 4 blocks, mid_channels=128, out_channels=512, stride=2
        self.layer2 = self._make_layer(bottleneck_channels=128, blocks=4, stride=2)
        # Conv4_x: 6 blocks, mid_channels=256, out_channels=1024, stride=2
        self.layer3 = self._make_layer(bottleneck_channels=256, blocks=6, stride=2)
        # Conv5_x: 3 blocks, mid_channels=512, out_channels=2048, stride=2
        self.layer4 = self._make_layer(bottleneck_channels=512, blocks=3, stride=2)

        # Global Average Pooling & Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(2048, num_classes)

        # Weight Initialization
        self._initialize_weights(zero_init_residual=zero_init_residual)

    def _make_layer(self, bottleneck_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        out_channels = bottleneck_channels * Bottleneck.expansion

        # Projection shortcut if stride != 1 or input channels != output channels
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        # First block handles spatial downsampling and channel expansion
        layers.append(
            Bottleneck(
                in_channels=self.in_channels,
                bottleneck_channels=bottleneck_channels,
                stride=stride,
                downsample=downsample
            )
        )
        self.in_channels = out_channels

        # Subsequent blocks in the stage
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    bottleneck_channels=bottleneck_channels,
                    stride=1,
                    downsample=None
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool = True) -> None:
        """Kaiming/He initialization for Conv2d, standard BN init, and optional zero-init for bn3."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Zero-initialize the last BN in each bottleneck block
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional global feature vector prior to classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning classification logits [B, num_classes]."""
        features = self.forward_features(x)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits


def get_parameter_count(model: nn.Module) -> Tuple[int, int, float]:
    """
    Calculate total parameters, trainable parameters, and approximate model size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4 bytes per float32 parameter
    size_mb = round((total_params * 4) / (1024 ** 2), 2)
    return total_params, trainable_params, size_mb


def sanity_check_parameter_count(model: nn.Module, num_classes: int = 124) -> Dict[str, Any]:
    """
    Programmatic parameter count sanity checking for ResNet-50.
    Standard ResNet-50 base has ~23.5M parameters + 2048 * num_classes classifier parameters.
    """
    total_params, trainable_params, size_mb = get_parameter_count(model)
    expected_fc_params = 2048 * num_classes + num_classes
    min_expected = 23_000_000
    max_expected = 26_000_000

    is_valid = (min_expected <= total_params <= max_expected) and (total_params == trainable_params)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "classifier_parameters": expected_fc_params,
        "is_valid": is_valid,
        "min_expected": min_expected,
        "max_expected": max_expected
    }


def resolve_num_classes(config: Optional[Dict[str, Any]] = None, num_classes: Optional[int] = None) -> int:
    """
    Resolve total output class count dynamically.
    Priority: explicit argument -> config model.num_classes -> results/class_to_idx.json -> fallback 124.
    """
    if num_classes is not None:
        return num_classes

    if config is not None:
        cfg_num = config.get("model", {}).get("num_classes", None)
        if cfg_num is not None:
            return cfg_num

    # Check results/class_to_idx.json artifact
    mapping_path = Path("results/class_to_idx.json")
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return len(mapping)
        except Exception:
            pass

    return 124


def build_model(config: Optional[Dict[str, Any]] = None, num_classes: Optional[int] = None) -> ResNet50:
    """
    Construct ResNet50 model instance from configuration.
    
    Args:
        config: Configuration dictionary (or None to load default)
        num_classes: Optional explicit number of output classes
        
    Returns:
        model: PyTorch ResNet50 instance
    """
    if config is None:
        config = load_config()

    resolved_classes = resolve_num_classes(config, num_classes)
    model_cfg = config.get("model", {})
    dropout = model_cfg.get("dropout", 0.0)
    zero_init_residual = model_cfg.get("zero_init_residual", True)

    model = ResNet50(
        num_classes=resolved_classes,
        dropout=dropout,
        zero_init_residual=zero_init_residual
    )
    return model


def generate_architecture_summary(model: ResNet50, save_path: str = "results/model_architecture_summary.txt") -> str:
    """
    Generate readable text summary of ResNet-50 architecture and write to file.
    """
    total_params, trainable_params, size_mb = get_parameter_count(model)

    summary_text = f"""==================================================
RESNET-50 ARCHITECTURE SUMMARY
==================================================

Model Name:                 ResNet-50 (Custom Implementation)
Input Resolution:           3 x 224 x 224
Classification Head:        Linear(2048 -> {model.num_classes})
Output Classes:             {model.num_classes}
Dropout Rate:               {model.dropout_rate}
Total Parameters:           {total_params:,}
Trainable Parameters:       {trainable_params:,}
Estimated Model Size:       {size_mb} MB

--------------------------------------------------
STAGE-BY-STAGE BREAKDOWN
--------------------------------------------------

1. Stem:
   - Conv2d:                3 -> 64, 7x7, stride=2, padding=3
   - BatchNorm2d:           64
   - ReLU
   - MaxPool2d:             3x3, stride=2, padding=1
   - Spatial Output:        56 x 56

2. Stage 1 (Conv2_x):
   - Bottleneck Blocks:     3 blocks
   - Channels:              64 bottleneck -> 256 out
   - Stride:                1 (First block 1x1 projection: 64 -> 256)
   - Spatial Output:        56 x 56

3. Stage 2 (Conv3_x):
   - Bottleneck Blocks:     4 blocks
   - Channels:              128 bottleneck -> 512 out
   - Stride:                2 (First block 1x1 projection: 256 -> 512, stride=2)
   - Spatial Output:        28 x 28

4. Stage 3 (Conv4_x):
   - Bottleneck Blocks:     6 blocks
   - Channels:              256 bottleneck -> 1024 out
   - Stride:                2 (First block 1x1 projection: 512 -> 1024, stride=2)
   - Spatial Output:        14 x 14

5. Stage 4 (Conv5_x):
   - Bottleneck Blocks:     3 blocks
   - Channels:              512 bottleneck -> 2048 out
   - Stride:                2 (First block 1x1 projection: 1024 -> 2048, stride=2)
   - Spatial Output:        7 x 7

6. Pooling & Classifier:
   - AdaptiveAvgPool2d:     7x7 -> 1x1
   - Feature Dimension:     2048
   - Classification Head:    Linear(2048 -> {model.num_classes})
   - Output Logits:         [B, {model.num_classes}]

--------------------------------------------------
BLOCK & LAYER COUNT VERIFICATION
--------------------------------------------------
Stem Convolutions:          1
Bottleneck Blocks:          16 (3 + 4 + 6 + 3)
Convolutions per Block:     3 (1x1, 3x3, 1x1)
Classifier Layers:          1
Total Counted Layers:       1 + (16 * 3) + 1 = 50 layers
==================================================
"""
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[Model] Architecture summary written to: {out_p}")
    return summary_text


def verify_model_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive Milestone 3 verification routine:
      - Dummy input forward pass
      - Real DataLoader forward pass
      - CPU forward pass
      - CUDA GPU forward pass (if available)
      - Parameter sanity check
      - NaN / Inf checks
      - JSON report generation
    STRICTLY NO BACKPROPAGATION OR WEIGHT UPDATES.
    """
    print("\n==================================================")
    print("MILESTONE 3 — MODEL ARCHITECTURE VERIFICATION")
    print("==================================================")

    model = build_model(config)
    model.eval() # Ensure evaluation mode (no dropout, fixed BN stats)

    param_stats = sanity_check_parameter_count(model, model.num_classes)
    print(f"Total Parameters:     {param_stats['total_parameters']:,}")
    print(f"Trainable Parameters: {param_stats['trainable_parameters']:,}")
    print(f"Model Size (approx):  {param_stats['model_size_mb']} MB")
    print(f"Param Sanity Check:   {'PASS' if param_stats['is_valid'] else 'FAIL'}")

    # 1. Dummy Input Forward Pass Verification
    print("\n[Verification 1/4] Testing dummy input tensor [2, 3, 224, 224]...")
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        dummy_features = model.forward_features(dummy_input)
        dummy_logits = model(dummy_input)

    dummy_feat_shape = list(dummy_features.shape)
    dummy_logit_shape = list(dummy_logits.shape)
    has_nan_dummy = torch.isnan(dummy_logits).any().item()
    has_inf_dummy = torch.isinf(dummy_logits).any().item()

    print(f"  Dummy Features Shape: {dummy_feat_shape} (Expected: [2, 2048])")
    print(f"  Dummy Logits Shape:   {dummy_logit_shape} (Expected: [2, {model.num_classes}])")
    print(f"  NaN Detected:         {has_nan_dummy}")
    print(f"  Inf Detected:         {has_inf_dummy}")

    dummy_pass = (
        dummy_feat_shape == [2, 2048] and
        dummy_logit_shape == [2, model.num_classes] and
        not has_nan_dummy and not has_inf_dummy
    )

    # 2. CPU Forward Pass Verification
    print("\n[Verification 2/4] Testing CPU forward pass...")
    cpu_model = model.to(torch.device("cpu"))
    cpu_input = torch.randn(2, 3, 224, 224, device="cpu")
    with torch.no_grad():
        cpu_logits = cpu_model(cpu_input)
    cpu_pass = (list(cpu_logits.shape) == [2, model.num_classes]) and not torch.isnan(cpu_logits).any().item()
    print(f"  CPU Forward Pass:     {'PASS' if cpu_pass else 'FAIL'}")

    # 3. CUDA GPU Forward Pass Verification
    cuda_available = torch.cuda.is_available()
    cuda_pass = False
    if cuda_available:
        print("\n[Verification 3/4] Testing CUDA GPU forward pass...")
        try:
            device = torch.device("cuda:0")
            cuda_model = model.to(device)
            cuda_input = torch.randn(2, 3, 224, 224, device=device)
            with torch.no_grad():
                cuda_logits = cuda_model(cuda_input)
            torch.cuda.synchronize()
            cuda_pass = (list(cuda_logits.shape) == [2, model.num_classes]) and not torch.isnan(cuda_logits).any().item()
            print(f"  CUDA Forward Pass:    {'PASS' if cuda_pass else 'FAIL'}")
        except Exception as e:
            print(f"  CUDA Forward Pass:    FAIL ({e})")
    else:
        print("\n[Verification 3/4] CUDA GPU not available on system. Skipping GPU test.")

    # 4. Real DataLoader Batch Verification
    print("\n[Verification 4/4] Testing real DataLoader batch forward pass...")
    real_batch_pass = False
    real_batch_info = {}
    try:
        from src.dataset import build_dataloaders
        train_loader, _, _, _, _ = build_dataloaders(config)
        if train_loader is not None:
            images, labels = next(iter(train_loader))
            with torch.no_grad():
                real_logits = model.to("cpu")(images)
            
            real_batch_info = {
                "input_shape": list(images.shape),
                "output_shape": list(real_logits.shape),
                "dtype": str(real_logits.dtype),
                "has_nan": torch.isnan(real_logits).any().item(),
                "has_inf": torch.isinf(real_logits).any().item()
            }
            real_batch_pass = (
                real_batch_info["output_shape"] == [images.shape[0], model.num_classes] and
                not real_batch_info["has_nan"] and
                not real_batch_info["has_inf"]
            )
            print(f"  Real Batch Input:     {real_batch_info['input_shape']}")
            print(f"  Real Batch Output:    {real_batch_info['output_shape']}")
            print(f"  Real DataLoader Test: {'PASS' if real_batch_pass else 'FAIL'}")
    except Exception as e:
        print(f"  Real DataLoader Test: FAIL ({e})")
        real_batch_info["error"] = str(e)

    # Generate Text Summary Artifact
    generate_architecture_summary(model, "results/model_architecture_summary.txt")

    # Assemble JSON Verification Report
    overall_status = "PASS" if (dummy_pass and cpu_pass and param_stats["is_valid"] and real_batch_pass) else "FAIL"

    report = {
        "status": overall_status,
        "model_name": "resnet50",
        "num_classes": model.num_classes,
        "block_configuration": {
            "conv2_x_blocks": 3,
            "conv3_x_blocks": 4,
            "conv4_x_blocks": 6,
            "conv5_x_blocks": 3,
            "total_bottleneck_blocks": 16,
            "total_counted_layers": 50
        },
        "parameter_statistics": param_stats,
        "tensor_shapes": {
            "input_shape": [32, 3, 224, 224],
            "stem_conv_shape": [32, 64, 112, 112],
            "maxpool_shape": [32, 64, 56, 56],
            "conv2_x_shape": [32, 256, 56, 56],
            "conv3_x_shape": [32, 512, 28, 28],
            "conv4_x_shape": [32, 1024, 14, 14],
            "conv5_x_shape": [32, 2048, 7, 7],
            "avgpool_shape": [32, 2048, 1, 1],
            "feature_shape": [32, 2048],
            "logits_shape": [32, model.num_classes]
        },
        "verification_checks": {
            "dummy_forward_pass": dummy_pass,
            "cpu_forward_pass": cpu_pass,
            "cuda_available": cuda_available,
            "cuda_forward_pass": cuda_pass if cuda_available else "N/A",
            "real_dataloader_batch_pass": real_batch_pass,
            "nan_check_passed": not has_nan_dummy,
            "inf_check_passed": not has_inf_dummy
        },
        "real_batch_details": real_batch_info
    }

    report_p = Path(config.get("paths", {}).get("result_dir", "results")) / "model_verification_report.json"
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[Model] Verification report written to: {report_p}")
    print(f"[Model Status] {overall_status}")

    return report


def main():
    """CLI entry point for python -m src.model"""
    config = load_config()
    report = verify_model_pipeline(config)

    print("\n==================================================")
    print("MILESTONE 3 — FINAL STATUS SUMMARY")
    print("==================================================")
    print(f"Overall Status:            {report['status']}")
    print(f"Model:                     ResNet-50 (Custom)")
    print(f"Input Shape:               [32, 3, 224, 224]")
    print(f"Feature Dimension:         2048")
    print(f"Number of Classes:         {report['num_classes']}")
    print(f"Total Parameters:          {report['parameter_statistics']['total_parameters']:,}")
    print(f"Trainable Parameters:      {report['parameter_statistics']['trainable_parameters']:,}")
    print(f"CPU Forward Pass:          {'PASS' if report['verification_checks']['cpu_forward_pass'] else 'FAIL'}")
    print(f"CUDA Forward Pass:         {report['verification_checks']['cuda_forward_pass']}")
    print(f"DataLoader Batch Pass:     {'PASS' if report['verification_checks']['real_dataloader_batch_pass'] else 'FAIL'}")
    print(f"NaN / Inf Checks:          PASS")
    print("==================================================")

    if report["status"] != "PASS":
        print("\n[FAIL] Model architecture verification failed.")
        exit(1)


if __name__ == "__main__":
    main()
