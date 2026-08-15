import unittest
import torch
import torch.nn as nn

from src.utils import load_config
from src.model import (
    Bottleneck,
    ResNet50,
    build_model,
    get_parameter_count,
    sanity_check_parameter_count
)
from src.dataset import build_dataloaders


class TestMilestone3ResNet50(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/config.yaml")

    def test_01_bottleneck_block_output(self):
        """1. Verify Bottleneck block output tensor dimensions."""
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )
        block = Bottleneck(in_channels=64, bottleneck_channels=64, stride=1, downsample=downsample)
        x = torch.randn(2, 64, 56, 56)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, torch.Size([2, 256, 56, 56]))

    def test_02_identity_shortcut(self):
        """2. Verify Bottleneck identity shortcut when channels & resolution match."""
        block = Bottleneck(in_channels=256, bottleneck_channels=64, stride=1)
        self.assertIsNone(block.downsample)
        x = torch.randn(2, 256, 56, 56)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, torch.Size([2, 256, 56, 56]))

    def test_03_projection_shortcut(self):
        """3. Verify projection shortcut when spatial resolution or channels change."""
        downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        block = Bottleneck(in_channels=256, bottleneck_channels=128, stride=2, downsample=downsample)
        self.assertIsNotNone(block.downsample)
        x = torch.randn(2, 256, 56, 56)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, torch.Size([2, 512, 28, 28]))

    def test_04_05_06_07_stage_block_counts(self):
        """4, 5, 6, 7. Verify block counts for Conv2_x, Conv3_x, Conv4_x, Conv5_x."""
        model = ResNet50(num_classes=124)
        self.assertEqual(len(model.layer1), 3, "Conv2_x must have 3 Bottleneck blocks")
        self.assertEqual(len(model.layer2), 4, "Conv3_x must have 4 Bottleneck blocks")
        self.assertEqual(len(model.layer3), 6, "Conv4_x must have 6 Bottleneck blocks")
        self.assertEqual(len(model.layer4), 3, "Conv5_x must have 3 Bottleneck blocks")
        
        total_blocks = len(model.layer1) + len(model.layer2) + len(model.layer3) + len(model.layer4)
        self.assertEqual(total_blocks, 16, "Total bottleneck blocks must equal 16")

    def test_08_feature_shape(self):
        """8. Verify forward_features outputs [B, 2048]."""
        model = ResNet50(num_classes=124)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            feats = model.forward_features(x)
        self.assertEqual(feats.shape, torch.Size([2, 2048]))

    def test_09_10_classifier_output_shape(self):
        """9 & 10. Verify classifier outputs [B, 124]."""
        model = ResNet50(num_classes=124)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, torch.Size([2, 124]))

    def test_11_cpu_forward_pass(self):
        """11. Verify CPU forward pass."""
        model = ResNet50(num_classes=124).to("cpu")
        model.eval()
        x = torch.randn(2, 3, 224, 224, device="cpu")
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, torch.Size([2, 124]))

    def test_12_cuda_forward_pass_if_available(self):
        """12. Verify CUDA forward pass if GPU is available."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = ResNet50(num_classes=124).to(device)
            model.eval()
            x = torch.randn(2, 3, 224, 224, device=device)
            with torch.no_grad():
                logits = model(x)
            torch.cuda.synchronize()
            self.assertEqual(logits.shape, torch.Size([2, 124]))

    def test_13_14_no_nan_no_inf(self):
        """13 & 14. Verify outputs contain no NaN or Inf values."""
        model = ResNet50(num_classes=124)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        self.assertFalse(torch.isnan(logits).any().item(), "Output contains NaN")
        self.assertFalse(torch.isinf(logits).any().item(), "Output contains Inf")

    def test_15_parameter_count_sanity(self):
        """15. Verify parameter count calculation and sanity bounds."""
        model = ResNet50(num_classes=124)
        stats = sanity_check_parameter_count(model, 124)
        self.assertTrue(stats["is_valid"], f"Parameter count {stats['total_parameters']} outside valid bounds")
        self.assertGreater(stats["total_parameters"], 23_500_000)
        self.assertLess(stats["total_parameters"], 26_000_000)

    def test_16_real_dataloader_batch_compatibility(self):
        """16. Verify compatibility with real batch from train_loader."""
        train_loader, _, _, _, _ = build_dataloaders(self.config)
        self.assertIsNotNone(train_loader)
        images, labels = next(iter(train_loader))
        
        model = ResNet50(num_classes=124)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        self.assertEqual(logits.shape, torch.Size([images.shape[0], 124]))

    def test_17_build_model_from_config(self):
        """17. Verify model construction using build_model(config)."""
        model = build_model(self.config)
        self.assertGreater(model.num_classes, 0)
        self.assertIsInstance(model, ResNet50)


if __name__ == "__main__":
    unittest.main()
